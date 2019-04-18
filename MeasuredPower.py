from __future__ import print_function,division
from collections import namedtuple


class MeasuredPower(object):
	"""
	"""
	def __init__(self, info, info_id1, info_id2):
		self.info = info
		self.info_id1 = info_id1
		self.info_id2 = info_id2


class MeasuredPower1D(MeasuredPower):
	"""
	Measured 1D power spectrum.

	This is basically the same as nbodykit binned statistic, but
	for convenience can access k and power simply as Pk.k and Pk.P.

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
		k, P, num_summands : array_like, (N,)

		nbk_binned_stat : None, nbodykit BinnedStatistic 
			Contains 1D power spectrum. Must be None if k, P, num_summands are specified.
		"""
		super(MeasuredPower1D, self).__init__(info, info_id1, info_id2)
		self.bstat = nbk_binned_stat

		if nbk_binned_stat is None:
			self.k = k
			self.P = P
			self.num_summands = num_summands
	
		else:
			assert k is None
			assert P is None
			assert num_summands is None

			assert len(self.bstat.power.shape) == 1
			# 1d power, save k, P, Nmodes for convenience/backward compatibility of code
			self.k = self.bstat.power['k']
			self.P = self.bstat.power['power'].real
			self.Nmodes = self.bstat.power['modes'].real



class MeasuredPower2D(MeasuredPower):
	"""
	Measured 2D power spectrum in Nk x Nmu bins.

	An instance has the following attributes:

	self.k2d : array_like, (Nk, Nmu)
		k2d[i,j] is the value of k in the i-th k bin and j-th mu bin.

	self.k : array_like, (Nk*Nmu, )
		Flattened copy of k2d. Useful for code that assumes 1D k bins.

	self.mu2d : array_like, (Nk, Nmu)
		mu2d[i,j] is the value of mu in the i-th k bin and j-th mu bin.

	self.mu : array_like, (Nk*Nmu, )
		Flattened copy of mu2d.

	Can always access original nbk_binned_stat, e.g.through Pk.bstat.power.k 
	and Pk.bstat.power.power.real.

	For multipoles, use Pk.bstat.poles.k and Pk.bstat.poles.power_0, 
	Pk.bstat.poles.power_2, etc.

	"""
	def __init__(self, nbk_binned_stat=None, info=None, info_id1=None, info_id2=None):
		"""
		Init with nbk_binned_stat which contains a measured 2D power spectrum.

		Parameters
		----------
		nbk_binned_stat : nbodykit BinnedStatistic 
			Contains measured 2D power spectrum. 
		"""
		super(MeasuredPower2D, self).__init__(info, info_id1, info_id2)
		self.bstat = nbk_binned_stat

		assert len(self.bstat.power.shape) == 2
		# 2d power P(k,mu). To use all 1d code, store the 2d power as reshaped 1d arrays.
		raise Exception('reshape 2d to 1d')
		# store Nk and Nmu
		#self.k = np.reshape(self.bstat.power[:,:], (Nk*Nmu,))





