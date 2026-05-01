import eventlet
eventlet.monkey_patch()

import csv
import time
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from os_ken.ofproto import ofproto_v1_3
from os_ken.lib import hub

class PredictiveTelemetryApp(app_manager.OSKenApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(PredictiveTelemetryApp, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.prev_stats = {} 
        self.queue_depths = {} 
        self.latencies = {}
        
        # Dynamic Polling Intervals for Phase 4 (RL Integration)
        self.polling_intervals = {} 
        
        # Complete 4-Feature CSV Header
        self.csv_file = open('telemetry_dataset.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestamp', 'switch_id', 'port_no', 
            'rx_mbps', 'tx_mbps', 'rx_loss', 'tx_loss', 
            'avg_queue_depth', 'latency_ms'
        ])
        
        self.monitor_thread = hub.spawn(self._monitor)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, CONFIG_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.info(f"Registered switch: {datapath.id}")
                self.datapaths[datapath.id] = datapath
                # Set to Intensive mode (1s) immediately to build the ML dataset
                self.polling_intervals[datapath.id] = 1  
        elif ev.state == CONFIG_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.info(f"Unregistered switch: {datapath.id}")
                del self.datapaths[datapath.id]
                if datapath.id in self.polling_intervals:
                    del self.polling_intervals[datapath.id]

    def _monitor(self):
        tick = 0
        while True:
            for dp in self.datapaths.values():
                interval = self.polling_intervals.get(dp.id, 1)
                # Only request stats if the tick matches the switch's assigned interval
                if tick % interval == 0:
                    self._request_port_stats(dp)
                    self._request_queue_stats(dp)
                    self._measure_latency(dp)
            
            hub.sleep(1) # Base tick rate
            tick += 1

    def _measure_latency(self, datapath):
        """Sends an Echo Request with the current timestamp embedded."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        payload = str(time.time()).encode('utf-8')
        req = parser.OFPEchoRequest(datapath, data=payload)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPEchoReply, [MAIN_DISPATCHER, CONFIG_DISPATCHER])
    def _echo_reply_handler(self, ev):
        """Calculates latency when the Echo Reply returns."""
        try:
            dpid = ev.msg.datapath.id
            send_time = float(ev.msg.data.decode('utf-8'))
            latency_ms = (time.time() - send_time) * 1000
            self.latencies[dpid] = latency_ms
        except ValueError:
            pass 

    def _request_port_stats(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    def _request_queue_stats(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        req = parser.OFPQueueStatsRequest(datapath, 0, ofproto.OFPP_ANY, ofproto.OFPQ_ALL)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPQueueStatsReply, MAIN_DISPATCHER)
    def _queue_stats_reply_handler(self, ev):
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        for stat in body:
            self.queue_depths[(dpid, stat.port_no)] = stat.tx_bytes

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        current_time = time.time()

        for stat in sorted(body, key=lambda stat: stat.port_no):
            if stat.port_no != ev.msg.datapath.ofproto.OFPP_LOCAL:
                port_key = (dpid, stat.port_no)
                
                rx_mbps, tx_mbps = 0.0, 0.0
                rx_loss_rate, tx_loss_rate = 0.0, 0.0
                queue_depth = self.queue_depths.get(port_key, 0)
                latency = self.latencies.get(dpid, 0.0) 

                if port_key in self.prev_stats:
                    prev = self.prev_stats[port_key]
                    time_diff = current_time - prev['time']
                    
                    if time_diff > 0: 
                        rx_mbps = ((stat.rx_bytes - prev['rx_bytes']) * 8) / time_diff / 1000000.0
                        tx_mbps = ((stat.tx_bytes - prev['tx_bytes']) * 8) / time_diff / 1000000.0

                        delta_rx_pkts = stat.rx_packets - prev['rx_packets']
                        delta_tx_pkts = stat.tx_packets - prev['tx_packets']
                        delta_rx_dropped = stat.rx_dropped - prev['rx_dropped'] + (stat.rx_errors - prev['rx_errors'])
                        delta_tx_dropped = stat.tx_dropped - prev['tx_dropped'] + (stat.tx_errors - prev['tx_errors'])

                        rx_total = delta_rx_pkts + delta_rx_dropped
                        tx_total = delta_tx_pkts + delta_tx_dropped

                        if rx_total > 0: rx_loss_rate = delta_rx_dropped / rx_total
                        if tx_total > 0: tx_loss_rate = delta_tx_dropped / tx_total

                self.prev_stats[port_key] = {
                    'time': current_time,
                    'rx_bytes': stat.rx_bytes, 'tx_bytes': stat.tx_bytes,
                    'rx_packets': stat.rx_packets, 'tx_packets': stat.tx_packets,
                    'rx_dropped': stat.rx_dropped, 'tx_dropped': stat.tx_dropped,
                    'rx_errors': stat.rx_errors, 'tx_errors': stat.tx_errors
                }

                self.logger.info(f"SW:{dpid} P:{stat.port_no} | RX:{rx_mbps:.2f}M | Loss:{rx_loss_rate:.4f} | Q:{queue_depth}B | Lat:{latency:.2f}ms")                
                
                self.csv_writer.writerow([
                    current_time, dpid, stat.port_no, 
                    round(rx_mbps, 4), round(tx_mbps, 4), 
                    round(rx_loss_rate, 6), round(tx_loss_rate, 6), 
                    queue_depth, round(latency, 4)
                ])
                self.csv_file.flush()