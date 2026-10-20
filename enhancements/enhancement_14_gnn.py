"""
Enhancement 14: Graph Neural Network — Fraud Ring Detection
============================================================
Detects coordinated fraud rings — groups of accounts working
together that single-transaction ML completely misses.

How it works:
  1. Build a transaction graph: nodes = accounts, edges = transactions
  2. Compute graph features: degree, centrality, community membership
  3. Flag suspicious clusters (fraud rings) using graph analytics
  4. Score each account by its network neighborhood risk

In production: Use PyTorch Geometric or DGL for real GNN layers.
Here we use networkx (already installed) + custom graph features.

Real-world fraud rings this catches:
  - Mule account networks (money laundering chains)
  - Coordinated fake merchant fraud
  - SIM-swap ring attacks
  - Round-tripping fraud
"""

import networkx as nx
import numpy as np
import pickle
import os
from datetime import datetime
from collections import defaultdict

# ── Transaction Graph Builder ──────────────────────────────────────────────
class TransactionGraph:
    """
    Builds and maintains a directed weighted graph of transactions.
    Node = account ID
    Edge = transaction (weighted by amount, annotated with features)
    """
    def __init__(self):
        self.G = nx.DiGraph()
        self.transaction_edges = []   # full edge history

    def add_transaction(self, sender_id, receiver_id, amount, features=None, is_fraud=None):
        """Add transaction as a directed edge in the graph."""
        # Add nodes with account attributes
        if not self.G.has_node(sender_id):
            self.G.add_node(sender_id, type='account', txn_count=0, total_sent=0,
                            total_received=0, fraud_count=0, risk_score=0.0)
        if not self.G.has_node(receiver_id):
            self.G.add_node(receiver_id, type='account', txn_count=0, total_sent=0,
                            total_received=0, fraud_count=0, risk_score=0.0)

        # Update node attributes
        self.G.nodes[sender_id]['txn_count']   += 1
        self.G.nodes[sender_id]['total_sent']  += amount
        self.G.nodes[receiver_id]['txn_count'] += 1
        if 'total_received' not in self.G.nodes[receiver_id]:
            self.G.nodes[receiver_id]['total_received'] = 0
        self.G.nodes[receiver_id]['total_received'] += amount

        if is_fraud:
            self.G.nodes[sender_id]['fraud_count']   += 1
            self.G.nodes[receiver_id]['fraud_count'] += 1

        # Add/update edge
        if self.G.has_edge(sender_id, receiver_id):
            self.G[sender_id][receiver_id]['weight']    += amount
            self.G[sender_id][receiver_id]['txn_count'] += 1
        else:
            self.G.add_edge(sender_id, receiver_id,
                            weight=amount, txn_count=1,
                            first_seen=datetime.now().isoformat())

        self.transaction_edges.append({
            'sender':   sender_id,
            'receiver': receiver_id,
            'amount':   amount,
            'features': features,
            'is_fraud': is_fraud
        })

    def get_node_count(self):
        return self.G.number_of_nodes()

    def get_edge_count(self):
        return self.G.number_of_edges()


# ── Graph Neural Network (networkx-based) ─────────────────────────────────
class GraphFraudDetector:
    """
    GNN-style fraud detector using graph structural features.

    Message passing simulation:
      Each node aggregates information from its neighborhood,
      exactly like real GNN message passing layers.

    Production upgrade path:
        import torch
        from torch_geometric.nn import GCNConv, SAGEConv
        # Use GraphSAGE for inductive learning on new accounts
    """

    def __init__(self, graph: TransactionGraph):
        self.tg = graph
        self.G  = graph.G
        self._community_map  = {}
        self._centrality_map = {}
        self._risk_scores    = {}

    # ── Graph Feature Extraction (GNN Layer 1: Node Features) ─────────────
    def compute_node_features(self, account_id):
        """
        Computes structural graph features for one account.
        These serve as node embeddings in a real GNN.
        """
        if not self.G.has_node(account_id):
            return self._default_node_features()

        G_undir = self.G.to_undirected()

        # Degree features
        in_deg  = self.G.in_degree(account_id)
        out_deg = self.G.out_degree(account_id)
        deg_ratio = out_deg / max(1, in_deg)   # >1 = mostly sending (suspicious)

        # Amount features
        node    = self.G.nodes[account_id]
        sent    = node.get('total_sent', 0)
        recv    = node.get('total_received', 0)
        amount_ratio = sent / max(1, recv)     # high = sending money out (mule)

        # Neighbor risk (message passing: aggregate neighbors)
        neighbors = list(self.G.neighbors(account_id)) + \
                    list(self.G.predecessors(account_id))
        if neighbors:
            neighbor_fraud = np.mean([
                self.G.nodes[n].get('fraud_count', 0) / max(1, self.G.nodes[n].get('txn_count', 1))
                for n in neighbors
            ])
            neighbor_degree = np.mean([G_undir.degree(n) for n in neighbors])
        else:
            neighbor_fraud  = 0.0
            neighbor_degree = 0.0

        # Centrality (how central/important is this node in fraud network?)
        try:
            centrality = nx.pagerank(self.G, max_iter=50).get(account_id, 0)
        except:
            centrality = 0.0

        # Clustering coefficient (part of tight cluster = fraud ring)
        try:
            clustering = nx.clustering(G_undir, account_id)
        except:
            clustering = 0.0

        # Cycle detection (A→B→C→A = round-tripping fraud)
        try:
            cycles = len(list(nx.simple_cycles(
                self.G.subgraph(list(self.G.neighbors(account_id)) + [account_id])
            )))
        except:
            cycles = 0

        return np.array([
            in_deg, out_deg, deg_ratio,
            sent, recv, amount_ratio,
            neighbor_fraud, neighbor_degree,
            centrality * 100,
            clustering,
            cycles,
            node.get('fraud_count', 0)
        ])

    def _default_node_features(self):
        return np.zeros(12)

    # ── Community Detection (GNN Layer 2: Neighborhood Aggregation) ───────
    def detect_communities(self):
        """
        Detect fraud rings using Louvain-style community detection.
        Communities with high internal transaction density = fraud rings.
        """
        G_undir = self.G.to_undirected()
        if G_undir.number_of_nodes() < 3:
            return {}

        try:
            # Greedy modularity communities (proxy for Louvain)
            communities = list(nx.community.greedy_modularity_communities(G_undir))
        except:
            # Fallback: connected components
            communities = list(nx.connected_components(G_undir))

        community_risks = {}
        for cid, community in enumerate(communities):
            members = list(community)
            subgraph = self.G.subgraph(members)

            # Risk score for this community
            total_fraud = sum(self.G.nodes[m].get('fraud_count', 0) for m in members)
            total_txns  = sum(self.G.nodes[m].get('txn_count',   1) for m in members)
            density     = nx.density(subgraph.to_undirected())
            fraud_rate  = total_fraud / max(1, total_txns)

            risk = (fraud_rate * 0.5 + density * 0.3 +
                    min(len(members)/20, 1.0) * 0.2)

            community_risks[cid] = {
                'community_id':  cid,
                'members':       members,
                'size':          len(members),
                'density':       round(density, 3),
                'fraud_rate':    round(fraud_rate, 3),
                'risk_score':    round(risk, 3),
                'is_fraud_ring': risk > 0.4 and len(members) >= 3
            }
            for m in members:
                self._community_map[m] = cid

        return community_risks

    # ── Account Risk Scoring ───────────────────────────────────────────────
    def score_account(self, account_id):
        """
        Full GNN-style risk score for an account.
        Combines: node features + community risk + transaction patterns.
        """
        communities = self.detect_communities()
        node_feats  = self.compute_node_features(account_id)

        # Community risk
        comm_id   = self._community_map.get(account_id, -1)
        comm_data = communities.get(comm_id, {})
        comm_risk = comm_data.get('risk_score', 0.0)
        is_ring   = comm_data.get('is_fraud_ring', False)

        # Node-level fraud indicators
        deg_ratio    = node_feats[2]    # out/in degree ratio
        amount_ratio = node_feats[5]    # sent/received ratio
        neigh_fraud  = node_feats[6]    # neighbor fraud rate
        centrality   = node_feats[8]    # pagerank centrality
        clustering   = node_feats[9]    # clustering coefficient
        cycles       = node_feats[10]   # round-trip cycles

        # Weighted risk score (GNN readout function)
        risk = (
            min(deg_ratio / 10, 1.0)      * 0.15 +
            min(amount_ratio / 5, 1.0)    * 0.15 +
            neigh_fraud                   * 0.25 +
            min(centrality / 10, 1.0)     * 0.15 +
            clustering                    * 0.10 +
            min(cycles / 3, 1.0)          * 0.10 +
            comm_risk                     * 0.10
        )

        risk = min(float(risk), 1.0)

        # Determine ring membership
        ring_info = None
        if is_ring and account_id in comm_data.get('members', []):
            ring_info = {
                'ring_id':   comm_id,
                'ring_size': comm_data['size'],
                'density':   comm_data['density'],
                'fraud_rate':comm_data['fraud_rate']
            }

        return {
            'account_id':    account_id,
            'graph_risk':    round(risk * 100, 2),
            'is_fraud_ring_member': is_ring,
            'ring_info':     ring_info,
            'network_stats': {
                'in_degree':       int(node_feats[0]),
                'out_degree':      int(node_feats[1]),
                'neighbor_fraud_rate': round(float(neigh_fraud)*100, 1),
                'centrality_score':    round(float(centrality), 3),
                'clustering':          round(float(clustering), 3),
                'round_trip_cycles':   int(cycles),
            },
            'risk_factors':  _graph_risk_factors(node_feats, is_ring, ring_info)
        }

    def get_fraud_rings(self):
        """Return all detected fraud ring communities."""
        communities = self.detect_communities()
        rings = [c for c in communities.values() if c['is_fraud_ring']]
        rings.sort(key=lambda x: x['risk_score'], reverse=True)
        return rings

    def get_graph_summary(self):
        rings = self.get_fraud_rings()
        return {
            'total_accounts':  self.G.number_of_nodes(),
            'total_edges':     self.G.number_of_edges(),
            'fraud_rings':     len(rings),
            'accounts_in_rings': sum(r['size'] for r in rings),
            'highest_risk_ring': rings[0] if rings else None,
            'top_rings':       rings[:3]
        }


def _graph_risk_factors(feats, is_ring, ring_info):
    factors = []
    if feats[2] > 3:
        factors.append(f"High out/in degree ratio ({feats[2]:.1f}x) — possible money mule")
    if feats[5] > 2:
        factors.append("Sends significantly more than receives — suspicious flow pattern")
    if feats[6] > 0.3:
        factors.append(f"Connected to high-fraud neighbors ({feats[6]*100:.0f}% fraud rate)")
    if feats[10] > 0:
        factors.append(f"{int(feats[10])} round-trip transaction cycles detected — possible round-tripping")
    if feats[9] > 0.5:
        factors.append("Part of tightly clustered transaction group")
    if is_ring and ring_info:
        factors.append(f"Member of suspected fraud ring ({ring_info['ring_size']} accounts, density={ring_info['density']})")
    return factors


# ── Global graph instance ──────────────────────────────────────────────────
_transaction_graph   = TransactionGraph()
_graph_fraud_detector = None

def get_graph_detector():
    global _graph_fraud_detector
    if _graph_fraud_detector is None:
        _graph_fraud_detector = GraphFraudDetector(_transaction_graph)
    return _graph_fraud_detector

def add_to_graph(sender_id, receiver_id, amount, features=None, is_fraud=False):
    _transaction_graph.add_transaction(sender_id, receiver_id, amount, features, is_fraud)

def get_account_graph_score(account_id):
    det = get_graph_detector()
    return det.score_account(account_id)

def get_graph_summary():
    det = get_graph_detector()
    return det.get_graph_summary()


if __name__ == "__main__":
    print("="*60)
    print("  Enhancement 14: GNN Fraud Ring Detection Demo")
    print("="*60)

    np.random.seed(42)
    tg  = TransactionGraph()
    det = GraphFraudDetector(tg)

    # Simulate a fraud ring: A→B→C→D→A (ring of 4 mule accounts)
    ring_accounts = ['MULE_A', 'MULE_B', 'MULE_C', 'MULE_D']
    for i, src in enumerate(ring_accounts):
        dst = ring_accounts[(i+1) % len(ring_accounts)]
        for _ in range(5):
            tg.add_transaction(src, dst, np.random.uniform(5000, 50000),
                               is_fraud=True)

    # Simulate legitimate accounts
    for i in range(20):
        src = f'LEGIT_{i}'
        dst = f'MERCHANT_{np.random.randint(0,5)}'
        tg.add_transaction(src, dst, np.random.uniform(100, 3000), is_fraud=False)

    print(f"\nGraph built: {tg.get_node_count()} accounts, {tg.get_edge_count()} transactions")

    summary = det.get_graph_summary()
    print(f"\nFraud Rings Detected: {summary['fraud_rings']}")
    print(f"Accounts in Rings:    {summary['accounts_in_rings']}")

    # Score a ring member
    score = det.score_account('MULE_A')
    print(f"\nMULE_A Graph Risk Score: {score['graph_risk']}%")
    print(f"Is Fraud Ring Member:    {score['is_fraud_ring_member']}")
    print(f"Risk Factors:")
    for rf in score['risk_factors']:
        print(f"  → {rf}")

    # Score a legit account
    score_l = det.score_account('LEGIT_1')
    print(f"\nLEGIT_1 Graph Risk Score: {score_l['graph_risk']}%")
    print(f"Is Fraud Ring Member:     {score_l['is_fraud_ring_member']}")
    print("\n✅ GNN Demo complete!")
