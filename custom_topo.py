from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info

class FatTreeTopo(Topo):
    def build(self):
        # --- 1. Create the 8 Switches ---
        # 2 Core Switches
        c1 = self.addSwitch('s1', protocols='OpenFlow13')
        c2 = self.addSwitch('s2', protocols='OpenFlow13')

        # 2 Aggregation Switches
        a1 = self.addSwitch('s3', protocols='OpenFlow13')
        a2 = self.addSwitch('s4', protocols='OpenFlow13')

        # 4 Edge Switches
        e1 = self.addSwitch('s5', protocols='OpenFlow13')
        e2 = self.addSwitch('s6', protocols='OpenFlow13')
        e3 = self.addSwitch('s7', protocols='OpenFlow13')
        e4 = self.addSwitch('s8', protocols='OpenFlow13')

        # --- 2. Link the Switches (Fat-Tree Redundancy) ---
        # Connect Core to Aggregation
        self.addLink(c1, a1)
        self.addLink(c1, a2)
        self.addLink(c2, a1)
        self.addLink(c2, a2)

        # Connect Aggregation to Edge
        self.addLink(a1, e1)
        self.addLink(a1, e2)
        self.addLink(a2, e3)
        self.addLink(a2, e4)

        # --- 3. Create and Link the 16 Hosts ---
        # 4 hosts per Edge Switch
        host_num = 1
        edges = [e1, e2, e3, e4]
        for edge in edges:
            for _ in range(4):
                host_name = f'h{host_num}'
                host = self.addHost(host_name)
                self.addLink(edge, host)
                host_num += 1

def run_network():
    topo = FatTreeTopo()
    
    info("*** Starting Mininet Network\n")
    # Link it to your os-ken controller running on the same machine
    net = Mininet(topo=topo, switch=OVSKernelSwitch, controller=RemoteController, build=False)
    net.addController('c0', controller=RemoteController, ip='127.0.0.1', port=6653)
    
    net.build()
    net.start()
    
    info("*** Network Ready. Opening CLI...\n")
    CLI(net)
    
    info("*** Stopping Network\n")
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    run_network()