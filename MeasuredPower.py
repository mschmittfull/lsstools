from __future__ import print_function,division
from collections import namedtuple


class MeasuredPower(object):
	"""
	This is basically the same as nbodykit binned statistic, but
	for convenience can access k and power simply as Pk.k and Pk.P
	if it is a 1d power spectrum.

	Otherwise, access through Pk.bstat.power.k and Pk.bstat.power.power.real,
	For multipoles, use Pk.bstat.poles.k and Pk.bstat.poles.power_0, 
	Pk.bstat.poles.power_2, etc.

	Also save some additional information of the fields whose power
	is measured.
	"""

	def __init__(self, 
		k=None, P=None, num_summands=None,
		nbk_binned_stat=None,
		info=None, info_id1=None, info_id2=None):
		"""
		Init either with k, P, num_summands, or with nbk_binned_stat.

		Parameters
		----------
		k, P, num_summands : numpy.ndarray

		nbk_binned_stat : nbodykit BinnedStatistic instance
		"""
		self.bstat = nbk_binned_stat

		if nbk_binned_stat is not None:
			assert k is None
			assert P is None
			assert num_summands is None

			if len(self.bstat.power.shape) == 1:
				# 1d power, save k, P, Nmodes for convenience/backward compatibility of code
				self.k = self.bstat.power['k']
				self.P = self.bstat.power['power'].real
				self.Nmodes = self.bstat.power['modes'].real

		else:
			self.k = k
			self.P = P
			self.num_summands = num_summands

		self.info = info
		self.info_id1 = info_id1
		self.info_id2 = info_id2

