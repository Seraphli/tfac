import threading


class OpThread(threading.Thread):
    def __init__(self, sess, queue, op):
        super(OpThread, self).__init__()
        self.daemon = True
        self.sess = sess
        self.queue = queue
        self.op = op

    def run(self):
        while self.queue.get() == "run":
            self.sess.run(self.op)


class SummaryThread(threading.Thread):
    def __init__(self, sess, queue, op, sw):
        super(SummaryThread, self).__init__()
        self.daemon = True
        self.sess = sess
        self.queue = queue
        self.op = op
        self.sw = sw
        self.step = 0

    def run(self):
        while self.queue.get() == "run":
            summary = self.sess.run(self.op)
            for s in summary:
                self.sw.add_summary(s, self.step)
            self.step += 1