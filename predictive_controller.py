import eventlet
eventlet.monkey_patch()

import time
import numpy as np
import torch
import torch.nn as nn
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from os_ken.ofproto import ofproto_v1_3
from os_ken.lib import hub
from stable_baselines3 import DQN

# --- 1. Load LSTM Architecture ---
class CongestionPredictor(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2):
        super(CongestionPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

class PredictiveActivationApp(app_manager.OSKenApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(PredictiveActivationApp, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.prev_stats = {}
        self.queue_depths = {}
        self.latencies = {}
        
        # --- 2. Initialize AI Models ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.lstm = CongestionPredictor().to(self.device)
        self.lstm.load_state_dict(torch.load('lstm_forecaster.pth', map_location=self.device, weights_only=True))
        self.lstm.eval()
        
        self.rl_agent = DQN.load("dqn_telemetry_agent")
        self.logger.info("✅ Closed-Loop System Online: LSTM & DQN Models Loaded.")

        # --- 3. Dynamic Tracking Variables ---
        self.polling_intervals = {} # dpid -> interval in seconds
        self.action_mapping = {0: 30, 1: 10, 2: 1} # 0=Low, 1=Medium, 2=High
        self.current_action_idx = {} 
        self.history_buffer = {} # dpid_port -> list of the last 5 feature sets
        
        self.monitor_thread = hub.spawn(self._monitor)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, CONFIG_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.datapaths[datapath.id] = datapath
                self.polling_intervals[datapath.id] = 30 # Default to heartbeat mode
                self.current_action_idx[datapath.id] = 0
        elif ev.state == CONFIG_DISPATCHER:
            if datapath.id in self.datapaths:
                del self.datapaths[datapath.id]

    def _monitor(self):
        tick = 0
        while True:
            for dp in self.datapaths.values():
                interval = self.polling_intervals.get(dp.id, 30)
                # Only request stats if the tick matches the RL agent's assigned interval
                if tick % interval == 0:
                    self._request_port_stats(dp)
                    self._request_queue_stats(dp)
                    self._measure_latency(dp)
            hub.sleep(1) 
            tick += 1

    def _measure_latency(self, datapath):
        parser = datapath.ofproto_parser
        payload = str(time.time()).encode('utf-8')
        req = parser.OFPEchoRequest(datapath, data=payload)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPEchoReply, [MAIN_DISPATCHER, CONFIG_DISPATCHER])
    def _echo_reply_handler(self, ev):
        try:
            dpid = ev.msg.datapath.id
            send_time = float(ev.msg.data.decode('utf-8'))
            self.latencies[dpid] = (time.time() - send_time) * 1000
        except ValueError:
            pass 

    def _request_port_stats(self, datapath):
        req = datapath.ofproto_parser.OFPPortStatsRequest(datapath, 0, datapath.ofproto.OFPP_ANY)
        datapath.send_msg(req)

    def _request_queue_stats(self, datapath):
        req = datapath.ofproto_parser.OFPQueueStatsRequest(datapath, 0, datapath.ofproto.OFPP_ANY, datapath.ofproto.OFPQ_ALL)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPQueueStatsReply, MAIN_DISPATCHER)
    def _queue_stats_reply_handler(self, ev):
        for stat in ev.msg.body:
            self.queue_depths[(ev.msg.datapath.id, stat.port_no)] = stat.tx_bytes

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        current_time = time.time()

        for stat in sorted(body, key=lambda stat: stat.port_no):
            if stat.port_no != ev.msg.datapath.ofproto.OFPP_LOCAL:
                port_key = (dpid, stat.port_no)
                
                rx_mbps, tx_mbps, rx_loss_rate, tx_loss_rate = 0.0, 0.0, 0.0, 0.0
                queue_depth = self.queue_depths.get(port_key, 0)
                latency = self.latencies.get(dpid, 0.0) 

                if port_key in self.prev_stats:
                    prev = self.prev_stats[port_key]
                    time_diff = current_time - prev['time']
                    
                    if time_diff > 0: 
                        rx_mbps = ((stat.rx_bytes - prev['rx_bytes']) * 8) / time_diff / 1000000.0
                        tx_mbps = ((stat.tx_bytes - prev['tx_bytes']) * 8) / time_diff / 1000000.0
                        
                        rx_total = (stat.rx_packets - prev['rx_packets']) + (stat.rx_dropped - prev['rx_dropped'])
                        if rx_total > 0: rx_loss_rate = (stat.rx_dropped - prev['rx_dropped']) / rx_total

                self.prev_stats[port_key] = {
                    'time': current_time,
                    'rx_bytes': stat.rx_bytes, 'tx_bytes': stat.tx_bytes,
                    'rx_packets': stat.rx_packets, 'rx_dropped': stat.rx_dropped,
                }

                # --- 4. The Brain: RL Telemetry Escalation ---
                # Rough normalization to keep inputs balanced for the Neural Network
                feature_vector = [rx_mbps/100.0, tx_mbps/100.0, rx_loss_rate, tx_loss_rate, queue_depth/10000.0, latency/100.0]
                
                if port_key not in self.history_buffer:
                    self.history_buffer[port_key] = []
                self.history_buffer[port_key].append(feature_vector)
                
                # Keep sliding window at 5 seconds
                if len(self.history_buffer[port_key]) > 5:
                    self.history_buffer[port_key].pop(0)
                    
                # If we have enough history, ask the AI what to do next
                if len(self.history_buffer[port_key]) == 5:
                    window_tensor = torch.tensor([self.history_buffer[port_key]], dtype=torch.float32).to(self.device)
                    
                    with torch.no_grad():
                        lstm_prob = self.lstm(window_tensor).item()
                        
                    current_idx = self.current_action_idx.get(dpid, 0)
                    rl_state = np.array([lstm_prob, rx_mbps/100.0, current_idx], dtype=np.float32)
                    
                    action, _ = self.rl_agent.predict(rl_state, deterministic=True)
                    new_interval = self.action_mapping[int(action)]
                    
                    # Apply the newly predicted interval back to the controller
                    self.polling_intervals[dpid] = new_interval
                    self.current_action_idx[dpid] = int(action)
                    
                    self.logger.info(f"SW:{dpid} P:{stat.port_no} | Util:{rx_mbps:.2f}M | LSTM Risk:{lstm_prob:.0%} | Next Poll:{new_interval}s")