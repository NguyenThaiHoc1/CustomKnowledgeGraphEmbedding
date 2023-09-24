class BaseScorer:

    def __init__(self, head, relation, tail, mode):
        self.head = head
        self.relation = relation
        self.tail = tail
        self.mode = mode

    def compute_score(self):
        raise NotImplementedError
