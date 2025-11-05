# swarm/coordinator.py
# Coordinator: receives heartbeats & detections from agents, fuses them into GLOBAL targets,
# runs Hungarian assignment so each agent tracks a different target, and sends
# an optional "spacing nudge" when agents crowd the same bearing.
#
# Works with UDP/JSON + agent messages. No GPS/poses needed.

import time
import math
from collections import defaultdict

from common import (
    COORD_LISTEN, make_rx_socket, make_tx_socket, recv_json, send_json
)

def now_ms() -> int:
    return int(time.time() * 1000)

def l2(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# ---------- target fusion across agents (shared X axis) ----------
class GlobalTargets:
    """
    Maintains a list of global targets by clustering detections that are
    close in the shared 'global_x' axis and have same class.
    """
    def __init__(self, max_age_ms=1000, gate_px=80):
        self.max_age_ms = max_age_ms
        self.gate_px = gate_px
        self.next_id = 1
        self.tracks = {}

    def update(self, all_reports):
        """
        all_reports: list of dicts with fields:
          {"aid": "A1", "cls": "person", "p":0.88,
           "gx": 512, "gy": 240, "cx": 310, "cy": 200, "box":[...], "ow":1280, "oh":720}
           (gx/gy are 'global' center, cx/cy & box are agent-local, for echoing back)
        Returns:
          clusters = list of:
            {"tid": int, "cls": str, "gx": float, "gy": float,
             "rep": {aid: det_dict_for_that_agent, ...}}
        """
        now = now_ms()

        for tid in list(self.tracks.keys()):
            if now - self.tracks[tid]["ts"] > self.max_age_ms:
                del self.tracks[tid]

        # 2) Greedy matching of incoming dets to existing tracks (1-D gate on gx by class)
        assignments = {}   # det_index -> tid
        used_tracks = set()
        for i, d in enumerate(all_reports):
            best, best_tid = 1e9, None
            for tid, tr in self.tracks.items():
                if tr["cls"] != d["cls"] or tid in used_tracks:
                    continue
                dist = abs(tr["gx"] - d["gx"])
                if dist < self.gate_px and dist < best:
                    best, best_tid = dist, tid
            if best_tid is not None:
                assignments[i] = best_tid
                used_tracks.add(best_tid)

        # 3) Update assigned tracks / create new tracks
        for i, d in enumerate(all_reports):
            if i in assignments:
                tid = assignments[i]
                self.tracks[tid]["gx"] = 0.7 * self.tracks[tid]["gx"] + 0.3 * d["gx"]
                self.tracks[tid]["gy"] = 0.7 * self.tracks[tid]["gy"] + 0.3 * d["gy"]
                self.tracks[tid]["p"]  = max(self.tracks[tid]["p"], d["p"])
                self.tracks[tid]["ts"] = now
            else:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = {"cls": d["cls"], "gx": d["gx"], "gy": d["gy"], "p": d["p"], "ts": now}
                assignments[i] = tid

        # 4) Build clusters with per-agent representatives
        clusters = defaultdict(lambda: {"rep": {}})
        for i, d in enumerate(all_reports):
            tid = assignments[i]
            clusters[tid]["tid"] = tid
            clusters[tid]["cls"] = self.tracks[tid]["cls"]
            clusters[tid]["gx"]  = self.tracks[tid]["gx"]
            clusters[tid]["gy"]  = self.tracks[tid]["gy"]
            clusters[tid]["rep"][d["aid"]] = d  # agent-local view
        return list(clusters.values())

# ---------- Hungarian (scipy) ----------
def hungarian(cost):
    # cost: list of lists; rectangular allowed (pad with BIG)
    # returns list of (i,j) chosen pairs where i=row(agent), j=col(target)
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception as e:
        raise SystemExit("Please install scipy: pip install scipy") from e
    import numpy as np
    cost = np.asarray(cost, dtype=float)
    if cost.size == 0:
        return []
    ri, cj = linear_sum_assignment(cost)
    return list(zip(ri.tolist(), cj.tolist()))

# ---------- main loop ----------
def main():
    rx = make_rx_socket(COORD_LISTEN)  # bind 127.0.0.1:9000
    tx = make_tx_socket()

    print(f"[Coordinator] listening on {COORD_LISTEN[0]}:{COORD_LISTEN[1]}")

    # Agent state
    # aid -> {"port": int, "last_hb": int(ms), "last_det": list[det dict]}
    agents = {}
    gt = GlobalTargets()

    TICK = 0.25   # assign at 4 Hz (smooth for gimbals), detections can be higher
    last_tick = 0.0

    while True:
        now = time.time()

        # ---- receive one pending UDP (non-blocking friendly) ----
        res = recv_json(rx)
        if res:
            msg, _addr = res
            if isinstance(msg, dict):
                t = msg.get("t")
                aid = msg.get("id")
                if t == "hb" and aid:
                    # Expect msg to contain {"listen_port": int}
                    port = msg.get("listen_port")
                    if isinstance(port, int) and port > 0:
                        agents.setdefault(aid, {"port": port, "last_hb": now_ms(), "last_det": []})
                        agents[aid]["port"] = port
                        agents[aid]["last_hb"] = now_ms()
                elif t == "det" and aid:
                    st = agents.get(aid)
                    # Only accept detections from agents we’ve seen a heartbeat for
                    if st and isinstance(st.get("port"), int):
                        dets = msg.get("d") or []
                        if isinstance(dets, list):
                            st["last_det"] = dets

        # ---- periodically compute assignments ----
        if now - last_tick >= TICK:
            last_tick = now

            # 1) collect all reports and build global target clusters
            all_reports = []
            valid_agent_ids = []
            for aid, st in agents.items():
                port = st.get("port")
                if not (isinstance(port, int) and port > 0):
                    continue
                valid_agent_ids.append(aid)
                for d in (st.get("last_det") or []):
                    # expect det dict like:
                    # {"c":"person","p":0.88,"cx":int,"cy":int,"gx":float,"gy":float,
                    #  "box":[x1,y1,x2,y2],"ow":w,"oh":h}
                    if isinstance(d, dict):
                        if all(k in d for k in ("c","p","cx","cy","gx","gy","box","ow","oh")):
                            all_reports.append({"aid": aid, "cls": d["c"], "p": float(d["p"]),
                                                "cx": int(d["cx"]), "cy": int(d["cy"]),
                                                "gx": float(d["gx"]), "gy": float(d["gy"]),
                                                "box": [int(x) for x in d["box"]],
                                                "ow": int(d["ow"]), "oh": int(d["oh"])})

            clusters = gt.update(all_reports) if all_reports else []

            # 2) build cost matrix (agents x clusters)
            BIG = 1e6
            agent_ids = valid_agent_ids
            costs = []
            if clusters and agent_ids:
                for aid in agent_ids:
                    row = []
                    for cl in clusters:
                        d = cl["rep"].get(aid)  # cost is finite only if this agent sees this cluster
                        if not d:
                            row.append(BIG)
                        else:
                            midx, midy = d["ow"]//2, d["oh"]//2
                            slew = abs(d["cx"] - midx) + abs(d["cy"] - midy)
                            slew_norm = slew / max(midx + midy, 1)
                            conf_penalty = (1.0 - float(d["p"]))
                            row.append(0.7 * slew_norm + 0.3 * conf_penalty)
                    costs.append(row if row else [BIG])

            # 3) Hungarian assignment
            pairs = []
            if clusters and agent_ids and any(any(c < BIG for c in row) for row in costs):
                pairs = hungarian(costs)

            # 4) Prepare per-agent replies
            for i, aid in enumerate(agent_ids):
                reply = {"t": "assign", "to": aid, "ts": now_ms(), "mode": "search"}

                chosen_j = None
                for (ri, cj) in pairs:
                    if ri == i and cj < len(costs[i]) and costs[i][cj] < BIG:
                        chosen_j = cj
                        break

                if chosen_j is not None and chosen_j < len(clusters):
                    cl = clusters[chosen_j]
                    d = cl["rep"].get(aid)
                    if d:
                        reply.update({
                            "mode": "track",
                            "target_id": int(cl["tid"]),
                            "target_cls": cl["cls"],
                            "local": {  # agent-local bearing info so it can draw and compute error
                                "cx": int(d["cx"]), "cy": int(d["cy"]),
                                "box": [int(x) for x in d["box"]],
                                "conf": float(d["p"]),
                                "ow": int(d["ow"]), "oh": int(d["oh"])
                            }
                        })
                        # --- spacing nudge (screen-space deconfliction) ---
                        try:
                            near_center = []
                            midx = reply["local"]["ow"] // 2
                            my_cx = reply["local"]["cx"]
                            for (ri, cj) in pairs:
                                if ri == i or cj >= len(clusters):
                                    continue
                                other_cl = clusters[cj]
                                other_det = other_cl["rep"].get(aid)  # how THIS agent sees that cluster
                                if other_det is None:
                                    continue
                                near_center.append(other_det["cx"])
                            nudge = 0.0
                            for ocx in near_center:
                                if abs(ocx - midx) < 0.08 * reply["local"]["ow"] and abs(my_cx - midx) < 0.08 * reply["local"]["ow"]:
                                    nudge = (+3.0 if my_cx <= ocx else -3.0)
                                    break
                            if nudge != 0.0:
                                reply["nudge"] = {"pan_rate_dps": nudge}
                        except Exception:
                            pass

                # 5) send (only if we have a valid port)
                port = agents.get(aid, {}).get("port")
                if isinstance(port, int) and port > 0:
                    try:
                        send_json(tx, ("127.0.0.1", port), reply)
                    except Exception as e:
                        print(f"[Coordinator] send to {aid}@{port} failed: {e}")


if __name__ == "__main__":
    main()
