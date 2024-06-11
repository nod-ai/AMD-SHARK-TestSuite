import pytest

def pytest_addoption(parser):
    parser.addoption("--goldentime-rocm-e2e-ms", action="store", default=1661.5, type=float, help="Golden time to test benchmark")
    parser.addoption("--goldentime-rocm-unet-ms", action="store", default=450.5, type=float, help="Golden time to test benchmark")
    parser.addoption("--goldentime-rocm-clip-ms", action="store", default=19, type=float, help="Golden time to test benchmark")
    parser.addoption("--goldentime-rocm-vae-ms", action="store", default=288.5, type=float, help="Golden time to test benchmark")
    parser.addoption("--gpu-number", action="store", default=6, type=int, help="IREE GPU device number to test on")
    parser.addoption("--rocm-chip", action="store", default="gfx90a", type=str, help="ROCm target chip configuration of GPU")

@pytest.fixture
def goldentime_rocm_e2e(request):
    return request.config.getoption("--goldentime-rocm-e2e-ms")

@pytest.fixture
def goldentime_rocm_unet(request):
    return request.config.getoption("--goldentime-rocm-unet-ms")

@pytest.fixture
def goldentime_rocm_clip(request):
    return request.config.getoption("--goldentime-rocm-clip-ms")

@pytest.fixture
def goldentime_rocm_vae(request):
    return request.config.getoption("--goldentime-rocm-vae-ms")

@pytest.fixture
def rocm_chip(request):
    return request.config.getoption("--rocm-chip")

@pytest.fixture
def gpu_number(request):
    return request.config.getoption("--gpu-number")
