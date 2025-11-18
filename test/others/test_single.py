import unittest
import numpy as np


class TestSinglePrecision(unittest.TestCase):
    def test_error_in_numerator_below_ulp(self):
        np.seterr(all="ignore")
        zero = np.float32(0.0)
        ulp_zero = np.nextafter(zero, np.float32(1.0))
        err = ulp_zero / np.float32(4.0)
        print("\n0_plus_err:", zero + err)
        print("1_div_err:", np.float32(1.0) / err)

    def test_error_in_denominator_below_ulp(self):
        np.seterr(all="ignore")
        zero = np.float32(0.0)
        ulp_zero = np.nextafter(zero, np.float32(1.0))
        err = ulp_zero / np.float32(4.0)
        print("\n0_plus_err:", zero + err)
        print("1_div_err:", np.float32(1.0) / err)


if __name__ == "__main__":
    #print("\nTestSinglePrecision:")
    #unittest.main(verbosity=2, failfast=False)

    print('\nfloat32:', np.float32(1e-8)*np.float32(1.0/4))