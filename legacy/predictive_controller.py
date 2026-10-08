import eventlet
eventlet.monkey_patch()

import time
import json
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from os_ken.ofproto import ofproto_v1_3
from os_ken.lib import hub
from kafka import KafkaProducer, KafkaConsumer

class PredictiveActivationApp(app_manager.OSKenApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(PredictiveActivationApp, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.prev_stats = {}
        self.queue_depths = {}
        self.latencies = {}
        self.polling_intervals = {} # dpid -> interval in seconds
        
        # Initialize Kafka Producer
        self.logger.info("Connecting Producer to Kafka...")
        self.producer = None
        for i in range(15):
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=['localhost:9092'],
                    value_serializer=lambda v: json.dumps(v).encode('utf-8')
                )
                self.logger.info("Kafka Producer connected successfully.")
                break
            except Exception as e:
                self.logger.warn(f"Waiting for Kafka Producer ({i+1}/15): {e}")
                hub.sleep(3)

        if not self.producer:
            self.logger.error("Failed to connect to Kafka. Exiting App.")
            return

        # Spawn threads
        self.monitor_thread = hub.spawn(self._monitor)
        self.control_consumer_thread = hub.spawn(self._consume_control_messages)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, CONFIG_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.info(f"Registered switch: {datapath.id}")
                self.datapaths[datapath.id] = datapath
                self.polling_intervals[datapath.id] = 30 # Default to heartbeat mode
                
                # Apply default routing path on Aggregation switches (s3, s4)
                if datapath.id in [3, 4]:
                    self._apply_routing_decision(datapath, "s1")
                    
        elif ev.state == CONFIG_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.info(f"Unregistered switch: {datapath.id}")
                del self.datapaths[datapath.id]
                if datapath.id in self.polling_intervals:
                    del self.polling_intervals[datapath.id]

    def _monitor(self):
        tick = 0
        while True:
            for dp in list(self.datapaths.values()):
                interval = self.polling_intervals.get(dp.id, 30)
                # Only request stats if the tick matches the switch's assigned interval
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

                # Publish Telemetry Message to Kafka
                payload = {
                    'timestamp': current_time,
                    'switch_id': dpid,
                    'port_no': stat.port_no,
                    'rx_mbps': round(rx_mbps, 4),
                    'tx_mbps': round(tx_mbps, 4),
                    'rx_loss': round(rx_loss_rate, 6),
                    'tx_loss': round(tx_loss_rate, 6),
                    'avg_queue_depth': queue_depth,
                    'latency_ms': round(latency, 4)
                }
                
                if self.producer:
                    self.producer.send('network-telemetry', value=payload)

    def _consume_control_messages(self):
        self.logger.info("Starting Kafka Control Consumer Thread...")
        consumer = None
        for i in range(15):
            try:
                consumer = KafkaConsumer(
                    'network-control',
                    bootstrap_servers=['localhost:9092'],
                    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
                )
                self.logger.info("Kafka Control Consumer connected successfully.")
                break
            except Exception as e:
                self.logger.warn(f"Waiting for Kafka Control Consumer ({i+1}/15): {e}")
                hub.sleep(3)
        
        if not consumer:
            self.logger.error("Failed to connect Control Consumer to Kafka.")
            return

        for message in consumer:
            decision = message.value
            try:
                dpid = int(decision['switch_id'])
                polling_interval = int(decision['polling_interval'])
                routing_path = decision['routing_path']
                
                # Apply dynamic polling rate
                self.polling_intervals[dpid] = polling_interval
                self.logger.info(f"Kafka Decision for SW {dpid} | Poll: {polling_interval}s | Route Path: {routing_path}")
                
                # Apply dynamic routing rules if the switch is currently registered
                if dpid in self.datapaths:
                    self._apply_routing_decision(self.datapaths[dpid], routing_path)
            except Exception as e:
                self.logger.error(f"Error handling Kafka control message: {e}")
            
            hub.sleep(0.01) # Yield to eventlet greenlets

    def _apply_routing_decision(self, datapath, path):
        """
        Dynamically updates flow rules on aggregation switches to reroute traffic.
        Aggregation switch s3 (dpid 3) connects to Core 1 (s1) on Port 1 and Core 2 (s2) on Port 2.
        Aggregation switch s4 (dpid 4) connects to Core 1 (s1) on Port 1 and Core 2 (s2) on Port 2.
        """
        if datapath.id not in [3, 4]:
            return
            
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto
        out_port = 1 if path == "s1" else 2
        
        self.logger.info(f"Updating Switch {datapath.id} flow rules: Routing all cross-network flows to Port {out_port} ({path})")
        
        # Priority 10 rules override the standard MAC-learning rules (which run at priority 1)
        if datapath.id == 3:
            # Route packets destined to h9-h16 (IPs 10.0.0.9 to 10.0.0.16)
            for host_id in range(9, 17):
                ip_dst = f"10.0.0.{host_id}"
                match = parser.OFPMatch(eth_type=0x0800, ipv4_dst=ip_dst)
                actions = [parser.OFPActionOutput(out_port)]
                self.add_flow(datapath, 10, match, actions)
                
        elif datapath.id == 4:
            # Route packets destined to h1-h8 (IPs 10.0.0.1 to 10.0.0.8)
            for host_id in range(1, 9):
                ip_dst = f"10.0.0.{host_id}"
                match = parser.OFPMatch(eth_type=0x0800, ipv4_dst=ip_dst)
                actions = [parser.OFPActionOutput(out_port)]
                self.add_flow(datapath, 10, match, actions)

    def add_flow(self, datapath, priority, match, actions):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=datapath, priority=priority,
            match=match, instructions=inst
        )
        datapath.send_msg(mod)