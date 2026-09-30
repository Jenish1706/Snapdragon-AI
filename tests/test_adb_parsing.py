from isingedge.bench.adb import Device, llama_cmd, parse_devices, parse_getprop


def test_parse_devices_handles_states_and_attributes():
    text = """List of devices attached
abc123   device usb:1-1 product:pad4 model:OPD2415 device:pad transport_id:3
xyz789   unauthorized usb:1-2 transport_id:4

"""
    rows = parse_devices(text)
    assert rows[0][0] == "abc123" and rows[0][1] == "device" and rows[0][2]["model"] == "OPD2415"
    assert rows[1][1] == "unauthorized"


def test_labels_by_soc_then_model():
    props = parse_getprop("[ro.soc.model]: [SM8850]\n[ro.product.model]: [OPD2415]\n")
    assert Device("s", "device", props).label == "pad4"
    assert Device("s", "device", {"ro.soc.model": "SM7550"}).label == "nordce4"
    assert Device("s", "device", {"ro.product.model": "Some Phone 9"}).label == "somephone9"


def test_llama_cmd_matches_qualcomm_runner_environment():
    cmd = llama_cmd("llama-bench", ["-m", "../models/x.gguf", "-dev", "HTP0"], "npu")
    assert "LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_DEVICES=HTP0" in cmd
    assert cmd.endswith("./bin/llama-bench -m ../models/x.gguf -dev HTP0")
    assert "GGML_HEXAGON_DEVICES" not in llama_cmd("llama-bench", [], "cpu")
