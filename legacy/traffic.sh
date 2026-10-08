h1 iperf -s -u &
h2 iperf -s &
h3 iperf -s -u &
h4 iperf -s &
h13 iperf -c 10.0.0.1 -u -b 100M -t 900 &
h14 iperf -c 10.0.0.3 -u -b 80M -t 900 &
h15 iperf -c 10.0.0.2 -t 900 -P 5 &
h16 iperf -c 10.0.0.4 -t 900 -P 10 &