# -*- coding: utf-8 -*-
"""
ERNIE Bot API Connection Test Script
Test Target: Verify text generation and image recognition
"""

import os
import sys
import base64

# Add project root to system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules from your project
from paddlebaidu.ernie_bot.base.ernie_bot_wrap import ErnieBotWrap, OrderPrompt, ActionPrompt


def test_text_model():
    """Test Text LLM Connection"""
    print("=" * 50)
    print("[TEST] Starting Text Model (ernie-4.5-turbo-128k) test...")
    print("=" * 50)

    # 1. Set environment variable to bypass YAML config reading
    os.environ["ERNIE_BOT_ACCESS_TOKEN"] = "YOUR_ERNIE_ACCESS_TOKEN"

    try:
        print("[INFO] Initializing client (Skipping YAML config)...")
        import erniebot
        from openai import OpenAI

        # Create object without calling __init__ to bypass yaml loading
        ernie_test = ErnieBotWrap.__new__(ErnieBotWrap)
        access_token = "YOUR_ERNIE_ACCESS_TOKEN"

        # Manually assign attributes (copied from your __init__)
        ernie_test.client = OpenAI(
            api_key=access_token,
            base_url="https://qianfan.baidubce.com/v2",
        )
        ernie_test.image_model = "ernie-4.5-turbo-vl"
        ernie_test.msgs = []
        ernie_test.model = 'ernie-4.5-turbo-128k'
        ernie_test.prompt_str = 'Please generate json based on the following description'

        print("[OK] Client initialized successfully!")

        # 2. Set Prompt and send request
        print("[INFO] Setting OrderPrompt and sending test query...")
        ernie_test.set_promt(str(OrderPrompt()))

                # Test query
        test_str = "Li Si in Building 2 wants to cook celery with meat, he needs celery now."
        
        # ?? �Ȼ�ȡԭʼ�ı���������ģ�͵���˵��ʲô�������ǲ��Ǳ�����
        state, str_res = ernie_test.get_res(test_str)
        print(f"[DEBUG] API Request State: {state}")
        print(f"[DEBUG] API Raw Response: {str_res}")
        
        # Ȼ���ٳ��Խ���
        json_res = ernie_test.get_json_str(str_res) if state else None

        # 3. Verify result
        if json_res and isinstance(json_res, dict) and 'name' in json_res:
            print("\n[SUCCESS] Text Model API call successful!")
            print("-" * 50)
            print(f"Model returned JSON: {json_res}")
            print(f"Extracted Name: {json_res['name']}")
            print(f"Extracted Goods: {json_res['goods']}")
            print(f"Extracted Address: {json_res['address']}")
            print("-" * 50)
            return True
        else:
            print("[WARNING] Model returned data, but parsing failed or format is incorrect:")
            print(json_res)
            return False

    except Exception as e:
        print(f"\n[FAILED] Text Model API call failed!")
        print(f"Error: {type(e).__name__}: {e}")
        return False


def test_image_model():
    """Test Vision LLM Connection"""
    print("\n" + "=" * 50)
    print("[TEST] Starting Vision Model (ernie-4.5-turbo-vl) test...")
    print("=" * 50)

    # Change this to an actual image path on your Jetson
    image_path = "./test_animal.jpg"

    if not os.path.exists(image_path):
        print(f"[SKIP] Test image {image_path} not found. Skipping vision model test.")
        print("Hint: Put an animal image named test_animal.jpg in the same directory to test.")
        return True  # Not a failure if no image

    try:
        print(f"[INFO] Reading image: {image_path}")
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode("utf-8")

        # Re-initialize client
        import erniebot
        from openai import OpenAI
        ernie_test = ErnieBotWrap.__new__(ErnieBotWrap)
        access_token = "YOUR_ERNIE_ACCESS_TOKEN"
        ernie_test.client = OpenAI(api_key=access_token, base_url="https://qianfan.baidubce.com/v2")
        ernie_test.image_model = "ernie-4.5-turbo-vl"
        ernie_test.msgs = []
        ernie_test.model = 'ernie-4.5-turbo-128k'
        ernie_test.prompt_str = ''

        print("[INFO] Sending image to ernie-4.5-turbo-vl model...")
        result, analysis = ernie_test.get_image_res(base64_image)

        print("\n[SUCCESS] Vision Model API call successful!")
        print("-" * 50)
        print(f"Analysis: {analysis}")
        print(f"Result (0=harmful, 1=beneficial): {result}")
        print("-" * 50)
        return True

    except Exception as e:
        print(f"\n[FAILED] Vision Model API call failed!")
        print(f"Error: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    print("[START] ERNIE Bot API Connection Test...\n")

    text_ok = test_text_model()
    image_ok = test_image_model()

    print("\n" + "=" * 50)
    print("[SUMMARY] Test Results:")
    print(f"  Text Model (ernie-4.5-turbo-128k): {'PASS' if text_ok else 'FAIL'}")
    print(f"  Vision Model (ernie-4.5-turbo-vl): {'PASS' if image_ok else 'FAIL'}")
    print("=" * 50)

    if text_ok:
        print("\n[INFO] Text model test passed. API key and network are working!")
        print("If your main program still crashes at yaml.safe_load, check config_car.yml path and format!")

    sys.exit(0 if text_ok and image_ok else 1)
