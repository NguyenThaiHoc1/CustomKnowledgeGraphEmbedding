class BaseScorer:

    def __init__(self, head, relation, tail, mode, W, mask):
        self.head = head
        self.relation = relation
        self.tail = tail
        self.W = W
        self.mask = mask
        self.mode = mode

    def compute_score(self):
        raise NotImplementedError
