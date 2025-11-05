# swarm/common.py
import json, socket, time
from dataclasses import dataclass, asdict

# Default ports
COORD_LISTEN = ("127.0.0.1", 9000)  # Coordinator binds here
# Agents pick their own reply port (e.g., 9101, 9102, ...)

# ---- Simple JSON-over-UDP helpers ----
def make_rx_socket(bind_addr):
    """bind_addr is (host, port) to bind for receiving"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(bind_addr)
    sock.setblocking(False)
    return sock

def make_tx_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    return sock

def send_json(sock, addr, obj):
    data = json.dumps(obj).encode("utf-8")
    sock.sendto(data, addr)

def recv_json(sock):
    try:
        data, addr = sock.recvfrom(65507)
        return json.loads(data.decode("utf-8")), addr
    except BlockingIOError:
        return None, None

def now_ms():
    return int(time.time() * 1000)

# ---- Message "schemas" (dicts) ----
# Agent -> Coordinator: heartbeat
def msg_heartbeat(agent_id, listen_port, pose=None, battery=100):
    return {
        "t": "hb",
        "id": agent_id,
        "ts": now_ms(),
        "listen_port": listen_port,
        "pose": pose or [0.0, 0.0, 0.0],  # x,y,z or lon,lat,alt if you have it
        "battery": battery
    }

# Agent -> Coordinator: detections
def msg_detections(agent_id, dets):
    # dets: list of {c: class, p: prob, cx, cy, box:[x1,y1,x2,y2]}
    return {
        "t": "det",
        "id": agent_id,
        "ts": now_ms(),
        "d": dets
    }

# Coordinator -> Agent: assignment
def msg_assignment(agent_id, note="track_best"):
    return {
        "t": "assign",
        "to": agent_id,
        "ts": now_ms(),
        "mode": note  # for demo, single mode: track_best
    }
