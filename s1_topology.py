from comnetsemu.net import Containernet
from mininet.node import Controller
from mininet.log import setLogLevel
from mininet.cli import CLI
import time

def create_topology():
    net = Containernet(controller=Controller)
    net.addController("c0")

    # Add switch
    s1 = net.addSwitch("s1")

    print("Adding Docker Hosts...")
    h1 = net.addDockerHost("h1", dimage="custom-traffic-generator",
                           ip="10.0.0.1", docker_args={"hostname": "h1"})
    print("h1 created successfully")

    h2 = net.addDockerHost("h2", dimage="tcpdump-capture",
                           ip="10.0.0.2", docker_args={"hostname": "h2"})
    print("h2 created successfully")

    # Connect hosts to the switch
    net.addLink(h1, s1)
    net.addLink(h2, s1)

    print("Starting network...")
    net.start()

    # Manually assign IP addresses to the correct interfaces
    print("Assigning IP addresses...")
    h1.cmd("ifconfig h1-eth0 10.0.0.1/24")
    h2.cmd("ifconfig h2-eth0 10.0.0.2/24")

    # Verify IP configurations
    print("Checking IP configurations...")
    h1_ip = h1.cmd("ifconfig h1-eth0")
    h2_ip = h2.cmd("ifconfig h2-eth0")
    print("h1 IP configuration:\n", h1_ip)
    print("h2 IP configuration:\n", h2_ip)

    print("Pinging h2 from h1...")
    ping_result = h1.cmd("ping -c 4 10.0.0.2")
    print(ping_result)

    # List all interfaces on h2
    print("Listing network interfaces on h2...")
    interfaces = h2.cmd("ip link show")
    print(interfaces)

    print("Starting traffic capture on h2...")
    # Run tcpdump in the foreground and save to a file
    tcpdump_process = h2.popen("tcpdump -w /app/capture.pcap")

    time.sleep(5)  # Give tcpdump some time to start capturing

    print("Generating traffic from h1...")
    traffic_result = h1.cmd("python3 /app/traffic_generator.py")
    print("Traffic generation result:\n", traffic_result)

    # Give some time for traffic generation
    time.sleep(80)

    print("Stopping tcpdump on h2...")
    tcpdump_process.terminate()

    CLI(net)
    net.stop()

if __name__ == "__main__":
    setLogLevel("info")
    create_topology()

