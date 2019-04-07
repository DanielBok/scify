import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from scify.specfunc import airy as a


@pytest.mark.parametrize('x, exp', [
    (np.arange(-3, 3.1, 0.4), [-0.378814293677658,
                               -0.178502428936404,
                               0.096145378007669,
                               0.340761559124214,
                               0.491700181061291,
                               0.535560883292352,
                               0.49484952543115,
                               0.406284187444801,
                               0.303703154286382,
                               0.209800061666379,
                               0.135292416312881,
                               0.0820380498076106,
                               0.0470362168668458,
                               0.0256104044217732,
                               0.0132892825296715,
                               0.00659113935746072]),
    (-2.35, -0.00833875172280131),
    (0.42, 0.250046304012225),
    (3.69, 0.00178056619302446)
])
def test_airy_Ai(x, exp):
    assert_almost_equal(a.airy_Ai(x), exp)


@pytest.mark.parametrize('x, exp', [
    (np.arange(-3, 5.1, 0.4), [0.314583769216598,
                               0.641637987111062,
                               0.686244824909002,
                               0.509997627719707,
                               0.239819119936297,
                               -0.0101605671166452,
                               -0.177362598696566,
                               -0.251032674005548,
                               -0.25240547028562,
                               -0.212793259389158,
                               -0.159147441296793,
                               -0.108509590480139,
                               -0.0685247801186109,
                               -0.0404972632444531,
                               -0.0225613108861087,
                               -0.0119129767059513,
                               -0.00598721902511379,
                               -0.00287375106090479,
                               -0.00132100066388768,
                               -0.000582914177810332,
                               -0.000247413890868462]),
    (-5.6, 0.850032560048932),
    (-2.35, 0.701094492082674),
    (-1.59, 0.371655902940422),
    (-0.42, -0.221288258720813),
    (3, -0.0119129767059513),
    (4.71, -0.000462278873173959)
])
def test_airy_Ai_deriv(x, exp):
    assert_almost_equal(a.airy_Ai_deriv(x), exp)


@pytest.mark.parametrize('x, exp', [
    (np.arange(-3, 3.1, 0.4), [-0.378814293677658,
                               -0.178502428936404,
                               0.096145378007669,
                               0.340761559124214,
                               0.491700181061291,
                               0.535560883292352,
                               0.49484952543115,
                               0.406284187444801,
                               0.322363321657436,
                               0.286000528176208,
                               0.26351364474914,
                               0.247526600878073,
                               0.235306006032376,
                               0.225521819439379,
                               0.217429335818205,
                               0.210572042785977]),
    (-2.35, -0.00833875172280131),
    (-0.42, 0.458689175879337),
    (0.42, 0.299797382951602),
    (3.69, 0.200825643616989)
])
def test_airy_Ai_scaled(x, exp):
    assert_almost_equal(a.airy_Ai_scaled(x), exp)


@pytest.mark.parametrize('x, exp', [
    (np.arange(-3, 5.1, 0.4), [0.314583769216598,
                               0.641637987111062,
                               0.686244824909002,
                               0.509997627719707,
                               0.239819119936297,
                               -0.0101605671166452,
                               -0.177362598696566,
                               -0.251032674005548,
                               -0.267913798909885,
                               -0.290080870778833,
                               -0.309976888960515,
                               -0.327396984170251,
                               -0.342805892948472,
                               -0.356613520770486,
                               -0.369131353043476,
                               -0.380592748019268,
                               -0.39117454043816,
                               -0.401013048884381,
                               -0.410215223380253,
                               -0.418866367099194,
                               -0.427035544351945]),
    (-5.6, 0.850032560048932),
    (-2.35, 0.701094492082674),
    (-1.59, 0.371655902940422),
    (-0.42, -0.221288258720813),
    (3, -0.380592748019268),
    (4.71, -0.421158335855092)
])
def test_airy_Ai_deriv_scaled(x, exp):
    assert_almost_equal(a.airy_Ai_deriv_scaled(x), exp)


@pytest.mark.parametrize('x, exp', [
    (np.arange(-3, 3.1, 0.4), [-0.198289626374927,
                               -0.405008278130034,
                               -0.450360984168208,
                               -0.341405831830135,
                               -0.134724060952795,
                               0.103997389496945,
                               0.32879184076087,
                               0.524509032818486,
                               0.705464202918661,
                               0.911063341694941,
                               1.20742359495287,
                               1.70365971153868,
                               2.59586935674391,
                               4.26703658176665,
                               7.51008769808228,
                               14.0373289637302]),
    (-2.35, -0.453321204001324),
    (0.42, 0.811984122611258),
    (1.77, 2.50837042480759),
    (3.69, 46.6905771821292)
])
def test_airy_Bi(x, exp):
    assert_almost_equal(a.airy_Bi(x), exp)


@pytest.mark.parametrize('x, exp', [
    (np.arange(-3, 3.1, 0.4), [-0.198289626374927,
                               -0.405008278130034,
                               -0.450360984168208,
                               -0.341405831830135,
                               -0.134724060952795,
                               0.103997389496945,
                               0.32879184076087,
                               0.524509032818486,
                               0.664628043168643,
                               0.668324448519242,
                               0.619911943572679,
                               0.564646061371301,
                               0.518898246927902,
                               0.484567448121886,
                               0.459016612762283,
                               0.439384023550096]),
    (-2.35, -0.453321204001324),
    (0.42, 0.677236161225369),
    (1.77, 0.521912888689588),
    (3.69, 0.413969360515798)
])
def test_airy_Bi_scaled(x, exp):
    assert_almost_equal(a.airy_Bi_scaled(x), exp)
