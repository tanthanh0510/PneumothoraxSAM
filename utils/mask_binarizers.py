class MaskBinarization:
    def __init__(self):
        self.thresholds = 0.5

    def transform(self, preds):
        yield preds > self.thresholds


class TripletMaskBinarization(MaskBinarization):
    def __init__(self, triplets, with_channels=False):
        super().__init__()
        self.thresholds = triplets
        self.dims = (2, 3) if with_channels else (1, 2)

    def transform(self, preds):
        for top_score_threshold, area_threshold, bottom_score_threshold in self.thresholds:
            clf_mask = preds > top_score_threshold
            preds_mask = preds > bottom_score_threshold
            preds_mask[clf_mask.sum(dim=self.dims) < area_threshold] = 0
            yield preds_mask


AVAI_MASK_BINARIZERS = {'TripletMaskBinarization': TripletMaskBinarization}


def get_mask_binarizes(cfg):
    name = cfg.name
    assert name in AVAI_MASK_BINARIZERS, print(
        '{name} is not supported, please implement it first.'.format(name=name))
    return AVAI_MASK_BINARIZERS[name](**cfg.params)
