from __future__ import division

from menpofit.atm.result import ATMAlgorithmResult, ATMResult


LucasKanadeAlgorithmResult = ATMAlgorithmResult


class LucasKanadeResult(ATMResult):
    r"""
    Class for storing the multi-scale iterative fitting result of a
    Lucas-Kanade alignment. It holds the shapes, shape parameters and costs
    per iteration.
    """
    def __str__(self):
        out = "LK alignment result of {} landmark points.".format(
                self.final_shape.n_points)
        if self.gt_shape is not None:
            out += "\nInitial error: {:.4f}".format(self.initial_error())
            out += "\nFinal error: {:.4f}".format(self.final_error())
        return out
