"""
Enhancement 12: Real-Time Transaction Streaming
================================================
Simulates Apache Kafka-style real-time streaming pipeline.
Uses Python threading + queue for the demo.

Architecture:
  Producer (generates transactions)
    → Stream Queue (Kafka topic proxy)
      → Consumer (fraud detection worker)
        → Results Store (processed transactions)

In production:
    pip install kafka-python confluent-kafka
    Replace StreamQueue with real Kafka producer/consumer
"""

import threading
import queue
import time
import uuid
import numpy as np
import pickle
import os
from datetime import datetime
from collections import deque

# ── Stream Queue (Kafka proxy) ─────────────────────────────────────────────
class StreamQueue:
    """
    In-memory message queue that mimics Kafka topic behavior.
    Production replacement:
        from kafka import KafkaProducer, KafkaConsumer
        producer = KafkaProducer(bootstrap_servers='localhost:9092')
        producer.send('upi-transactions', json.dumps(msg).encode())
    """
    def __init__(self, maxsize=1000):
        self._q = queue.Queue(maxsize=maxsize)
        self.total_published = 0
        self.total_consumed  = 0

    def publish(self, message):
        """Publish message to stream (like Kafka producer.send)"""
        self._q.put(message, timeout=1)
        self.total_published += 1

    def consume(self, timeout=0.5):
        """Consume next message (like Kafka consumer.poll)"""
        try:
            msg = self._q.get(timeout=timeout)
            self.total_consumed += 1
            return msg
        except queue.Empty:
            return None

    @property
    def pending(self):
        return self._q.qsize()


# ── Streaming Fraud Detector ───────────────────────────────────────────────
class StreamingFraudDetector:
    """
    Real-time fraud detection worker.
    Runs as a background thread, processing transactions as they arrive.
    """
    def __init__(self, stream_queue, result_store, model=None):
        self.stream    = stream_queue
        self.results   = result_store
        self.model     = model
        self._running  = False
        self._thread   = None

        # Live metrics (rolling window)
        self.processed_count  = 0
        self.fraud_count      = 0
        self.avg_latency_ms   = 0
        self._latencies       = deque(maxlen=100)
        self.throughput_log   = []   # (timestamp, tps)
        self._last_tps_check  = time.time()
        self._tps_count       = 0

    def _process_transaction(self, txn):
        """Core detection logic — runs per message."""
        start = time.time()

        features = np.array([[
            txn['amount'], txn['hour'], txn['frequency'],
            txn['distance'], txn['failed_attempts'],
            txn['new_device'], txn['vpn'], txn['ben_age'],
            txn['same_bank'], txn['weekend']
        ]])

        if self.model:
            prob = float(self.model.predict_proba(features)[0][1])
        else:
            # Heuristic fallback
            score = 0.0
            if txn['hour'] < 6:          score += 0.25
            if txn['new_device']:        score += 0.20
            if txn['vpn']:               score += 0.18
            if txn['failed_attempts']>2: score += 0.20
            if txn['ben_age'] < 5:       score += 0.15
            if txn['amount'] > 20000:    score += 0.10
            prob = min(score, 0.99)

        latency_ms = (time.time() - start) * 1000
        self._latencies.append(latency_ms)
        self.avg_latency_ms = np.mean(self._latencies)

        if prob >= 0.70:   decision = 'BLOCKED';      risk = 'HIGH'
        elif prob >= 0.40: decision = 'OTP_REQUIRED'; risk = 'MEDIUM'
        else:              decision = 'APPROVED';     risk = 'LOW'

        result = {
            'txn_id':     txn['txn_id'],
            'account_id': txn['account_id'],
            'amount':     txn['amount'],
            'probability':round(prob*100, 2),
            'decision':   decision,
            'risk':       risk,
            'latency_ms': round(latency_ms, 2),
            'timestamp':  datetime.now().strftime('%H:%M:%S.%f')[:-3],
            'stream_lag': txn.get('_stream_lag_ms', 0)
        }

        self.results.append(result)
        self.processed_count += 1
        self._tps_count += 1
        if prob >= 0.50: self.fraud_count += 1

        # Measure throughput every second
        now = time.time()
        if now - self._last_tps_check >= 1.0:
            tps = self._tps_count / (now - self._last_tps_check)
            self.throughput_log.append({'time': datetime.now().strftime('%H:%M:%S'),
                                        'tps': round(tps, 1)})
            if len(self.throughput_log) > 30:
                self.throughput_log.pop(0)
            self._tps_count = 0
            self._last_tps_check = now

        return result

    def _worker(self):
        """Background thread: continuously polls stream and processes."""
        while self._running:
            msg = self.stream.consume(timeout=0.1)
            if msg:
                self._process_transaction(msg)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print("🚀 Stream consumer started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        print("🛑 Stream consumer stopped")

    @property
    def metrics(self):
        return {
            'processed':    self.processed_count,
            'fraud_caught': self.fraud_count,
            'fraud_rate':   round(self.fraud_count / max(1, self.processed_count) * 100, 1),
            'avg_latency':  round(self.avg_latency_ms, 2),
            'pending':      self.stream.pending,
            'throughput':   self.throughput_log[-5:] if self.throughput_log else []
        }


# ── Transaction Producer (simulates live UPI transactions) ─────────────────
class TransactionProducer:
    """
    Generates realistic transaction streams.
    Production: replace with actual UPI gateway webhook events.
    """
    def __init__(self, stream_queue, fraud_rate=0.05):
        self.stream     = stream_queue
        self.fraud_rate = fraud_rate
        self._running   = False
        self._thread    = None
        self.published  = 0

    def _generate_transaction(self):
        is_fraud = np.random.random() < self.fraud_rate
        txn_time = time.time()

        if is_fraud:
            return {
                'txn_id':        'TXN' + str(uuid.uuid4())[:8].upper(),
                'account_id':    'ACC' + str(np.random.randint(10000, 99999)),
                'amount':        float(np.random.choice([
                                    np.random.uniform(1, 50),
                                    np.random.uniform(15000, 50000)
                                 ])),
                'hour':          int(np.random.choice(range(0, 6))),
                'frequency':     int(np.random.randint(20, 60)),
                'distance':      float(np.random.uniform(500, 5000)),
                'failed_attempts':int(np.random.randint(3, 8)),
                'new_device':    1,
                'vpn':           1,
                'ben_age':       int(np.random.randint(0, 5)),
                'same_bank':     0,
                'weekend':       1,
                '_stream_lag_ms':round((time.time() - txn_time) * 1000, 2),
                '_is_fraud':     True
            }
        else:
            return {
                'txn_id':        'TXN' + str(uuid.uuid4())[:8].upper(),
                'account_id':    'ACC' + str(np.random.randint(10000, 99999)),
                'amount':        float(np.random.uniform(50, 5000)),
                'hour':          int(np.random.randint(8, 22)),
                'frequency':     int(np.random.randint(1, 10)),
                'distance':      float(np.random.uniform(0, 50)),
                'failed_attempts':int(np.random.randint(0, 1)),
                'new_device':    0,
                'vpn':           0,
                'ben_age':       int(np.random.randint(30, 1000)),
                'same_bank':     1,
                'weekend':       0,
                '_stream_lag_ms':round((time.time() - txn_time) * 1000, 2),
                '_is_fraud':     False
            }

    def _producer_loop(self, rate_per_sec=5):
        interval = 1.0 / rate_per_sec
        while self._running:
            txn = self._generate_transaction()
            try:
                self.stream.publish(txn)
                self.published += 1
            except queue.Full:
                pass  # Drop if queue full (like Kafka backpressure)
            time.sleep(interval)

    def start(self, rate_per_sec=5):
        self._running = True
        self._thread = threading.Thread(
            target=self._producer_loop,
            args=(rate_per_sec,),
            daemon=True
        )
        self._thread.start()
        print(f"📡 Producer started — generating {rate_per_sec} txn/sec")

    def stop(self):
        self._running = False


# ── Global streaming pipeline ─────────────────────────────────────────────
_stream_queue = StreamQueue(maxsize=500)
_result_store = deque(maxlen=200)
_detector     = None
_producer     = None

def get_stream_pipeline():
    global _detector, _producer
    if _detector is None:
        model = None
        if os.path.exists('models/fraud_model.pkl'):
            model = pickle.load(open('models/fraud_model.pkl', 'rb'))
        _detector = StreamingFraudDetector(_stream_queue, _result_store, model)
        _producer = TransactionProducer(_stream_queue)
    return _detector, _producer, _stream_queue, _result_store

def start_streaming():
    det, prod, _, _ = get_stream_pipeline()
    if not det._running:
        det.start()
        prod.start(rate_per_sec=3)
        return True
    return False

def stop_streaming():
    det, prod, _, _ = get_stream_pipeline()
    det.stop(); prod.stop()

def get_stream_metrics():
    det, prod, q, results = get_stream_pipeline()
    return {
        'metrics':   det.metrics,
        'published': prod.published,
        'recent':    list(results)[-20:][::-1],
        'running':   det._running
    }


if __name__ == "__main__":
    print("="*60)
    print("  Enhancement 12: Real-Time Streaming Demo")
    print("="*60)

    det, prod, q, results = get_stream_pipeline()
    det.start()
    prod.start(rate_per_sec=10)

    print("\nStreaming for 5 seconds...\n")
    for i in range(5):
        time.sleep(1)
        m = det.metrics
        print(f"  t+{i+1}s | Processed: {m['processed']:3d} | "
              f"Fraud: {m['fraud_caught']:2d} ({m['fraud_rate']}%) | "
              f"Latency: {m['avg_latency']:.1f}ms | "
              f"Queue: {m['pending']}")

    det.stop(); prod.stop()
    print("\n✅ Streaming demo complete!")
