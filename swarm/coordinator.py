# swarm/coordinator.py
# Receives heartbeats & detections from agents; sends back simple assignments.

import time
from common import (
    COORD_LISTEN, make_rx_socket, make_tx_socket, recv_json, send_json, msg_assignment
)

def main():
    rx = make_rx_socket(COORD_LISTEN)  # bind 127.0.0.1:9000
    tx = make_tx_socket()

    agents = {}  # agent_id -> {"port": int, "last_hb": ms, "last_det": list}
    print(f"[Coordinator] listening on {COORD_LISTEN[0]}:{COORD_LISTEN[1]}")

    while True:
        msg, addr = recv_json(rx)
        now = int(time.time() * 1000)

        # Periodically print alive agents
        if int(time.time()) % 5 == 0:
            alive = [aid for aid, st in agents.items() if (now - st["last_hb"]) < 4000]
            # (This prints many times; it's fine for a demo.)
            print(f"[Coordinator] alive agents: {alive}", end="\r")

        if not msg:
            time.sleep(0.01)
            continue

        t = msg.get("t")
        aid = msg.get("id", "unknown")

        if t == "hb":
            listen_port = int(msg.get("listen_port", 0))
            st = agents.get(aid, {"port": listen_port, "last_hb": now, "last_det": []})
            st["port"] = listen_port
            st["last_hb"] = now
            agents[aid] = st
            # Optional: print first time we see the agent
            if listen_port and (now - st["last_hb"] < 100):  # crude "first sight" heuristic
                print(f"\n[Coordinator] heartbeat from {aid} @ {listen_port}")

        elif t == "det":
            st = agents.get(aid)
            if not st:
                # Unknown agent; ignore until we get a heartbeat
                continue
            st["last_det"] = msg.get("d", [])
            # For demo, send a simple assignment back: "track_best"
            reply = msg_assignment(aid, note="track_best")
            send_json(tx, ("127.0.0.1", st["port"]), reply)

        # Quit on Ctrl+C in terminal
    # never reached

if __name__ == "__main__":
    main()
