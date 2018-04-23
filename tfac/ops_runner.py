import threading
from queue import Queue


class Cmd(object):
    EXEC = "exec"
    EXIT = "exit"


class OpThread(threading.Thread):
    def __init__(self, queue, op):
        super(OpThread, self).__init__()
        self.daemon = True
        self.cmd_queue = queue
        self.op = op

    def prepare(self, sess):
        self.sess = sess

    def run(self):
        cmd = self.cmd_queue.get()
        while cmd != Cmd.EXIT:
            if cmd == Cmd.EXEC:
                self.sess.run(self.op)
            cmd = self.cmd_queue.get()


class OpRunner(object):
    def __init__(self, op):
        self.queue = Queue()
        self.thread = OpThread(self.queue, op)

    def run(self, sess):
        self.sess = sess
        self.thread.prepare(sess)
        self.thread.start()

    def execute(self):
        self.queue.put(Cmd.EXEC)

    def join(self):
        self.queue.join()

    def close(self):
        self.queue.put(Cmd.EXIT)
        self.thread.join()


class SummaryThread(threading.Thread):
    def __init__(self, cmd_queue, summary_op, global_step, summary_writer):
        super(SummaryThread, self).__init__()
        self.daemon = True
        self.cmd_queue = cmd_queue
        self.summary_op = summary_op
        self.global_step = global_step
        self.summary_writer = summary_writer

    def prepare(self, sess):
        self.sess = sess

    def run(self):
        cmd = self.cmd_queue.get()
        while cmd != Cmd.EXIT:
            if cmd == Cmd.EXEC:
                summary, g_step = self.sess.run(
                    [self.summary_op, self.global_step])
                for s in summary:
                    self.summary_writer.add_summary(s, g_step)
            cmd = self.cmd_queue.get()


class SummaryRunner(object):
    def __init__(self, summary_op, global_step, summary_writer):
        self.queue = Queue()
        self.thread = SummaryThread(self.queue, summary_op,
                                    global_step, summary_writer)

    def run(self, sess):
        self.sess = sess
        self.thread.prepare(sess)
        self.thread.start()

    def execute(self):
        self.queue.put(Cmd.EXEC)

    def close(self):
        self.queue.put(Cmd.EXIT)
        self.thread.join()
