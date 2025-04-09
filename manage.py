#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car

Usage:
    manage.py (drive) [--model=<model>] [--model2=<model2>] [--js] [--type=(linear|categorical)] [--camera=(single|stereo)] [--meta=<key:value> ...] [--myconfig=<filename>]
    manage.py (train) [--tubs=tubs] (--model=<model>) [--type=(linear|inferred|tensorrt_linear|tflite_linear)]

Options:
    -h --help               Show this screen.
    --js                    Use physical joystick.
    -f --file=<file>        A text file containing paths to tub files, one per line. Option may be used more than once.
    --meta=<key:value>      Key/Value strings describing describing a piece of meta data about this drive. Option may be used more than once.
    --myconfig=filename     Specify myconfig file to use. 
                            [default: myconfig.py]
"""
###add^^^[--model2=<model2>]
###add^^^[--duo=<None/'y'>]

from docopt import docopt

#
# import cv2 early to avoid issue with importing after tensorflow
# see https://github.com/opencv/opencv/issues/14884#issuecomment-599852128
#
try:
    import cv2
except:
    pass


import donkeycar as dk
from donkeycar.parts.transform import TriggeredCallback, DelayedTrigger
from donkeycar.parts.tub_v2 import TubWriter
from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.controller import LocalWebController, WebFpv, JoystickController
from donkeycar.parts.throttle_filter import ThrottleFilter
from donkeycar.parts.behavior import BehaviorPart
from donkeycar.parts.file_watcher import FileWatcher
from donkeycar.parts.launch import AiLaunch
from donkeycar.parts.kinematics import NormalizeSteeringAngle, UnnormalizeSteeringAngle, TwoWheelSteeringThrottle
from donkeycar.parts.kinematics import Unicycle, InverseUnicycle, UnicycleUnnormalizeAngularVelocity
from donkeycar.parts.kinematics import Bicycle, InverseBicycle, BicycleUnnormalizeAngularVelocity
from donkeycar.parts.explode import ExplodeDict
from donkeycar.parts.transform import Lambda
from donkeycar.parts.pipe import Pipe
from donkeycar.utils import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


###def drive(cfg, model_path=None, use_joystick=False, model_type=None,
###          camera_type='single', meta=[]):
def drive(cfg, model_path=None,model_path2=None,duo=None, use_joystick=False, model_type=None,
          camera_type='single', meta=[]):
    """
    Construct a working robotic vehicle from many parts. Each part runs as a
    job in the Vehicle loop, calling either it's run or run_threaded method
    depending on the constructor flag `threaded`. All parts are updated one
    after another at the framerate given in cfg.DRIVE_LOOP_HZ assuming each
    part finishes processing in a timely manner. Parts may have named outputs
    and inputs. The framework handles passing named outputs to parts
    requesting the same named input.
    """
    logger.info(f'PID: {os.getpid()}')
    if cfg.DONKEY_GYM:
        #the simulator will use cuda and then we usually run out of resources
        #if we also try to use cuda. so disable for donkey_gym.
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    if model_type is None:
        if cfg.TRAIN_LOCALIZER:
            model_type = "localizer"
        elif cfg.TRAIN_BEHAVIORS:
            model_type = "behavior"
        else:
            model_type = cfg.DEFAULT_MODEL_TYPE

    # Initialize car
    V = dk.vehicle.Vehicle()

    # Initialize logging before anything else to allow console logging
    if cfg.HAVE_CONSOLE_LOGGING:
        logger.setLevel(logging.getLevelName(cfg.LOGGING_LEVEL))
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(cfg.LOGGING_FORMAT))
        logger.addHandler(ch)

    if cfg.HAVE_MQTT_TELEMETRY:
        from donkeycar.parts.telemetry import MqttTelemetry
        tel = MqttTelemetry(cfg)
        
    #
    # if we are using the simulator, set it up
    #
    add_simulator(V, cfg)


    #
    # setup encoders, odometry and pose estimation
    #
    add_odometry(V, cfg)

    ###
    add_opticalflow(V, cfg)

    #
    # setup primary camera
    #
    add_camera(V, cfg, camera_type)


    # add lidar
    if cfg.USE_LIDAR:
        from donkeycar.parts.lidar import RPLidar
        if cfg.LIDAR_TYPE == 'RP':
            print("adding RP lidar part")
            lidar = RPLidar(lower_limit = cfg.LIDAR_LOWER_LIMIT, upper_limit = cfg.LIDAR_UPPER_LIMIT)
            V.add(lidar, inputs=[],outputs=['lidar/dist_array'], threaded=True)
        if cfg.LIDAR_TYPE == 'YD':
            print("YD Lidar not yet supported")
    ### add TFMINI 2D lidar
    elif cfg.HAVE_TMINI:
        from tmini import TMINI
        #from donkeycar.parts.tfmini import TFMini
        lidar = TMINI(cfg.ZONE_INDEX)
        V.add(lidar, inputs=[], outputs=['lidar/dist', 'lidar/wall_info', 'lidar/image_array'], threaded=False)
    ### add GS2 2D lidar 
    elif cfg.HAVE_GS2:
        import gs2
        lidar = gs2.GS2(cfg.ZONE_INDEX)
        V.add(lidar, inputs=[], outputs=['lidar/dist'], threaded=False)

    if cfg.SHOW_FPS:
        from donkeycar.parts.fps import FrequencyLogger
        V.add(FrequencyLogger(cfg.FPS_DEBUG_INTERVAL),
              outputs=["fps/current", "fps/fps_list"])


    #
    # add the user input controller(s)
    # - this will add the web controller
    # - it will optionally add any configured 'joystick' controller
    #
    has_input_controller = hasattr(cfg, "CONTROLLER_TYPE") and cfg.CONTROLLER_TYPE != "mock"
    ctr = add_user_controller(V, cfg, use_joystick)

    #
    # convert 'user/steering' to 'user/angle' to be backward compatible with deep learning data
    #
    V.add(Pipe(), inputs=['user/steering'], outputs=['user/angle'])

    #
    # explode the buttons input map into individual output key/values in memory
    #
    V.add(ExplodeDict(V.mem, "web/"), inputs=['web/buttons'])

    #
    # For example: adding a button handler is just adding a part with a run_condition
    # set to the button's name, so it runs when button is pressed.
    #
    V.add(Lambda(lambda v: print(f"web/w1 clicked")), inputs=["web/w1"], run_condition="web/w1")
    V.add(Lambda(lambda v: print(f"web/w2 clicked")), inputs=["web/w2"], run_condition="web/w2")
    V.add(Lambda(lambda v: print(f"web/w3 clicked")), inputs=["web/w3"], run_condition="web/w3")
    V.add(Lambda(lambda v: print(f"web/w4 clicked")), inputs=["web/w4"], run_condition="web/w4")
    V.add(Lambda(lambda v: print(f"web/w5 clicked")), inputs=["web/w5"], run_condition="web/w5")

    #this throttle filter will allow one tap back for esc reverse
    th_filter = ThrottleFilter()
    V.add(th_filter, inputs=['user/throttle'], outputs=['user/throttle'])

    #
    # maintain run conditions for user mode and autopilot mode parts.
    #
    V.add(UserPilotCondition(show_pilot_image=getattr(cfg, 'SHOW_PILOT_IMAGE', False)),
          inputs=['user/mode', "cam/image_array", "cam/image_array_trans"],
          outputs=['run_user', "run_pilot", "ui/image_array"])

    class LedConditionLogic:
        def __init__(self, cfg):
            self.cfg = cfg

        def run(self, mode, recording, recording_alert, behavior_state, model_file_changed, track_loc):
            #returns a blink rate. 0 for off. -1 for on. positive for rate.

            if track_loc is not None:
                led.set_rgb(*self.cfg.LOC_COLORS[track_loc])
                return -1

            if model_file_changed:
                led.set_rgb(self.cfg.MODEL_RELOADED_LED_R, self.cfg.MODEL_RELOADED_LED_G, self.cfg.MODEL_RELOADED_LED_B)
                return 0.1
            else:
                led.set_rgb(self.cfg.LED_R, self.cfg.LED_G, self.cfg.LED_B)

            if recording_alert:
                led.set_rgb(*recording_alert)
                return self.cfg.REC_COUNT_ALERT_BLINK_RATE
            else:
                led.set_rgb(self.cfg.LED_R, self.cfg.LED_G, self.cfg.LED_B)

            if behavior_state is not None and model_type == 'behavior':
                r, g, b = self.cfg.BEHAVIOR_LED_COLORS[behavior_state]
                led.set_rgb(r, g, b)
                return -1 #solid on

            if recording:
                return -1 #solid on
            elif mode == 'user':
                return 1
            elif mode == 'local_angle':
                return 0.5
            elif mode == 'local':
                return 0.1
            return 0

    if cfg.HAVE_RGB_LED and not cfg.DONKEY_GYM:
        from donkeycar.parts.led_status import RGB_LED
        led = RGB_LED(cfg.LED_PIN_R, cfg.LED_PIN_G, cfg.LED_PIN_B, cfg.LED_INVERT)
        led.set_rgb(cfg.LED_R, cfg.LED_G, cfg.LED_B)

        V.add(LedConditionLogic(cfg), inputs=['user/mode', 'recording', "records/alert", 'behavior/state', 'modelfile/modified', "pilot/loc"],
              outputs=['led/blink_rate'])

        V.add(led, inputs=['led/blink_rate'])

    def get_record_alert_color(num_records):
        col = (0, 0, 0)
        for count, color in cfg.RECORD_ALERT_COLOR_ARR:
            if num_records >= count:
                col = color
        return col

    class RecordTracker:
        def __init__(self):
            self.last_num_rec_print = 0
            self.dur_alert = 0
            self.force_alert = 0

        def run(self, num_records):
            if num_records is None:
                return 0

            if self.last_num_rec_print != num_records or self.force_alert:
                self.last_num_rec_print = num_records

                if num_records % 10 == 0:
                    print("recorded", num_records, "records")

                if num_records % cfg.REC_COUNT_ALERT == 0 or self.force_alert:
                    self.dur_alert = num_records // cfg.REC_COUNT_ALERT * cfg.REC_COUNT_ALERT_CYC
                    self.force_alert = 0

            if self.dur_alert > 0:
                self.dur_alert -= 1

            if self.dur_alert != 0:
                return get_record_alert_color(num_records)

            return 0

    rec_tracker_part = RecordTracker()
    V.add(rec_tracker_part, inputs=["tub/num_records"], outputs=['records/alert'])

    if cfg.AUTO_RECORD_ON_THROTTLE:
        def show_record_count_status():
            rec_tracker_part.last_num_rec_print = 0
            rec_tracker_part.force_alert = 1
        if (cfg.CONTROLLER_TYPE != "pigpio_rc") and (cfg.CONTROLLER_TYPE != "MM1"):  # these controllers don't use the joystick class
            if isinstance(ctr, JoystickController):
                ctr.set_button_down_trigger('circle', show_record_count_status) #then we are not using the circle button. hijack that to force a record count indication
        else:
            
            show_record_count_status()

    #Sombrero
    if cfg.HAVE_SOMBRERO:
        from donkeycar.parts.sombrero import Sombrero
        s = Sombrero()

    #IMU
    add_imu(V, cfg)


    # Use the FPV preview, which will show the cropped image output, or the full frame.
    if cfg.USE_FPV:
        V.add(WebFpv(), inputs=['cam/image_array'], threaded=True)

    def load_model(kl, model_path):
        start = time.time()
        print('loading model', model_path)
        kl.load(model_path)
        print('finished loading in %s sec.' % (str(time.time() - start)) )

    ###model2
    def load_model2(kl, model_path,model_path2):
        start = time.time()
        print('loading model', model_path)
        print('loading model2', model_path2)
        kl.load2(model_path,model_path2)
        print('finished loading in %s sec.' % (str(time.time() - start)) )
    ###

    def load_weights(kl, weights_path):
        start = time.time()
        try:
            print('loading model weights', weights_path)
            kl.model.load_weights(weights_path)
            print('finished loading in %s sec.' % (str(time.time() - start)) )
        except Exception as e:
            print(e)
            print('ERR>> problems loading weights', weights_path)

    def load_model_json(kl, json_fnm):
        start = time.time()
        print('loading model json', json_fnm)
        from tensorflow.python import keras
        try:
            with open(json_fnm, 'r') as handle:
                contents = handle.read()
                kl.model = keras.models.model_from_json(contents)
            print('finished loading json in %s sec.' % (str(time.time() - start)) )
        except Exception as e:
            print(e)
            print("ERR>> problems loading model json", json_fnm)

    ### add for behavior recording
    if cfg.TRAIN_BEHAVIORS:
        bh = BehaviorPart(cfg.BEHAVIOR_LIST)
        V.add(bh, outputs=['behavior/state', 'behavior/label', "behavior/one_hot_state_array"])
        try:
            ctr.set_button_down_trigger('L1', bh.increment_state)
        except:
            pass

        inputs = ['cam/image_array', "behavior/one_hot_state_array"]


    ### add color filter logic
    if cfg.HAVE_COLOR_FILTER:
        from PIL import Image
        import numpy as np
        import cv2
        from logging import getLogger
        class ColorFilter:
            def __init__(self, roi_crop_top, roi_crop_bottom, color_filter, num_logic_array, num_logic_pixel, lower_color, upper_color):
                self.roi_crop_top = roi_crop_top
                self.roi_crop_bottom = roi_crop_bottom
                self.color_filter = color_filter
                self.num_logic_array = num_logic_array
                self.num_logic_pixel = num_logic_pixel
                self.lower_color = lower_color
                self.upper_color = upper_color
                self.one_hot_state_array = np.zeros(num_logic_array)

                logger = getLogger("color_filter")
                logger.info(f'Created with color_filter')
            
            def cut_image(self, image):
                pixels = np.array(image)
                pixels = pixels[self.roi_crop_top:self.roi_crop_bottom, :]
                #print("cut_image", pixels)
                return Image.fromarray(pixels)

            def apply_filter(self, image, no_filter):
                hsv_image = image.convert('HSV')
                hsv_array = np.array(hsv_image)
                mask = cv2.inRange(hsv_array, np.array(self.lower_color[no_filter]), np.array(self.upper_color[no_filter]))
                filtered_image = cv2.bitwise_and(np.array(image), np.array(image), mask=mask)
                #print("apply_filter", filtered_image.shape)
                return filtered_image
                #return Image.fromarray(filtered_image)
            
            def state_logic(self, img_arr, no_filter):
                # マスクされた画像のピクセル数をカウント
                masked_image_array = np.array(img_arr)
                non_black_pixels = np.count_nonzero(masked_image_array)
                #print("non_black@", no_filter, non_black_pixels)
                # ピクセル数が閾値以上の場合、one_hot_state_arrayを更新
                if non_black_pixels > self.num_logic_pixel:
                    self.one_hot_state_array[no_filter] = 1
                    print("Detected @",no_filter)
                else:
                    self.one_hot_state_array[no_filter] = 0

                return self.one_hot_state_array

            """
                def convertImageArrayToPILImage(self, img_arr):
                img = Image.fromarray(img_arr.astype('uint8'), 'RGB')
                return img
            """
            
            def run(self, img_arr):
                img_arr = self.cut_image(img_arr)
                self.one_hot_state_array = np.zeros(self.num_logic_array)
                for no_filter in range(self.num_logic_array):
                    img_arr_filtered = self.apply_filter(img_arr, no_filter)
                    self.one_hot_state_array = self.state_logic(img_arr_filtered, no_filter)
                #print(self.one_hot_state_array)
                return self.one_hot_state_array

        filter = ColorFilter(cfg.ROI_CROP_TOP, cfg.ROI_CROP_BOTTOM,
                             cfg.COLOR_FILTER, cfg.NUM_LOGIC_ARRARY, cfg.NUM_LOCIG_PIXEL,
                             cfg.LOWER_FILTER, cfg.UPPER_FILTER)
        V.add(filter,
              inputs=['cam/image_array'],
              outputs=['filter/one_hot_state_array'])#,
              #run_condition='run_pilot')
    ###
    #
    # load and configure model for inference
    #
    # if model_path:
    #     ###add@20230819 resnetmodel 
    #     #ResNet18
    #     if cfg.DEFAULT_AI_FRAMEWORK == 'pytorch' and'.ckpt' in model_path: #'.pth' :
    #         from donkeycar.parts.pytorch.torch_utils import get_model_by_type
    #         model_type = cfg.DEFAULT_MODEL_TYPE
    #         kl = get_model_by_type(model_type, cfg, checkpoint_path=model_path)
    #         #ResNet18(input_shape=input_shape)

    #         ###add pure pytorch model with timm models by romihi@20250407　 
    #         #pure pytorch models
    #     elif cfg.DEFAULT_AI_FRAMEWORK == 'pytorch' and '.pth' in model_path:
    #         import torch
    #         try:
    #             from torch2trt import TRTModule
    #             has_trt = True
    #         except ImportError:
    #             has_trt = False
            
    #         # CUDA利用可能性を確認
    #         use_cuda = torch.cuda.is_available()
    #         device = torch.device('cuda' if use_cuda else 'cpu')
            
    #         from model_catalog import get_model, load_model_weights
            
    #         # TensorRTモデルパスを設定（元のモデルパスから派生可能に）
    #         trt_model_path = os.path.splitext(model_path)[0] + '_trt.pth'
            
    #         # TensorRTモデルが利用可能かチェック
    #         use_trt = has_trt and use_cuda and os.path.exists(trt_model_path)
            
    #         try:
    #             if use_trt:
    #                 print(f"Loading TensorRT optimized model from {trt_model_path}")
    #                 kl = TRTModule()
    #                 kl.load_state_dict(torch.load(trt_model_path))
    #                 print("TensorRT model loaded successfully")
    #             else:
    #                 # TensorRTが使えない理由をログ出力
    #                 if not has_trt:
    #                     print("TensorRT not available: package not installed")
    #                 elif not use_cuda:
    #                     print("TensorRT not available: CUDA not available")
    #                 elif not os.path.exists(trt_model_path):
    #                     print(f"TensorRT model not found at {trt_model_path}")
                    
    #                 print(f"Loading standard PyTorch model from {model_path}")
    #                 kl = get_model(model_type, pretrained=False, input_size=(cfg.IMAGE_W, cfg.IMAGE_H))
    #                 kl = load_model_weights(kl, model_path, device)
                
    #             # モデルをデバイスに移動
    #             kl = kl.to(device)
    #             # 推論モードに設定
    #             kl.eval()
                    
    #         except Exception as e:
    #             print(f"Error loading model: {e}")
    #             print(f"Attempting to load with alternative method...")
    #             try:
    #                 # 代替方法でモデル読み込み試行
    #                 kl = get_model(model_type, pretrained=False, input_size=(cfg.IMAGE_W, cfg.IMAGE_H))
    #                 checkpoint = torch.load(model_path, map_location=device)
    #                 if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    #                     kl.load_state_dict(checkpoint['model_state_dict'])
    #                 else:
    #                     kl.load_state_dict(checkpoint)
    #                 kl = kl.to(device)
    #                 kl.eval()
    #                 print("Model loaded with alternative method")
    #             except Exception as e2:
    #                 print(f"All loading attempts failed: {e2}")
    #                 raise

    #         #シンプルな画像入力のみ
    #         inputs = ['cam/image_array']

    if model_path:
        ###add@20230819 resnetmodel 
        #ResNet18
        if cfg.DEFAULT_AI_FRAMEWORK == 'pytorch' and'.ckpt' in model_path: #'.pth' :
            from donkeycar.parts.pytorch.torch_utils import get_model_by_type
            model_type = cfg.DEFAULT_MODEL_TYPE
            kl = get_model_by_type(model_type, cfg, checkpoint_path=model_path)
            #ResNet18(input_shape=input_shape)

        ###add ONNX model support by romihi@20250408
        ###PRi5で5V3Aだと落ちる
        elif cfg.DEFAULT_AI_FRAMEWORK == 'onnx' and '.onnx' in model_path:
            import onnxruntime as ort
            import numpy as np
            from PIL import Image
            import torchvision.transforms as transforms
            
            print(f"Loading ONNX model from {model_path}")
            try:
                # 利用可能なプロバイダの表示
                providers = ort.get_available_providers()
                print(f"Available providers: {providers}")
                
                # プロバイダの優先順位を設定（高速なものを優先）
                provider_options = []
                if 'CUDAExecutionProvider' in providers:
                    provider_options.append('CUDAExecutionProvider')
                if 'OpenCLExecutionProvider' in providers:
                    provider_options.append('OpenCLExecutionProvider')
                provider_options.append('CPUExecutionProvider')  # 必ずCPUをフォールバックとして含める
                
                # ONNXランタイムセッションを作成
                ort_session = ort.InferenceSession(model_path, providers=provider_options)
                
                # 入力情報を取得
                input_name = ort_session.get_inputs()[0].name
                input_shape = ort_session.get_inputs()[0].shape
                
                # 入力シェイプからサイズを取得（通常は[batch, channels, height, width]）
                if len(input_shape) == 4:
                    height, width = input_shape[2], input_shape[3]
                else:
                    # デフォルトのサイズ
                    height, width = cfg.IMAGE_H, cfg.IMAGE_W
                    
                print(f"ONNX model input shape: {input_shape}, using size: {height}x{width}")
                
                # 前処理パイプラインを作成
                preprocess = transforms.Compose([
                    transforms.Resize((height, width)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                
                # ONNXモデルのラッパークラスを作成
                class ONNXModel:
                    def __init__(self, session, input_name, preprocess):
                        self.session = session
                        self.input_name = input_name
                        self.preprocess = preprocess
                        self._device = "ONNX Runtime"
                        self.name = "onnx_model"
                    
                    def run(self, img_arr, other_arr=None):
                        """
                        Donkeycar parts interface to run the part in the loop.
                        """
                        # PILイメージに変換して前処理を適用
                        pil_image = Image.fromarray(img_arr)
                        tensor_image = self.preprocess(pil_image)
                        
                        # バッチ次元を追加して正しい形状にする
                        np_image = tensor_image.numpy()[np.newaxis, ...]
                        
                        # ONNX Runtimeで推論を実行
                        outputs = self.session.run(None, {self.input_name: np_image})
                        
                        # 出力は通常、最初の要素にあります
                        result = outputs[0].reshape(-1)
                        
                        # 必要に応じて出力を[-1, 1]の範囲に正規化
                        # ONNXモデルの出力範囲によって調整が必要な場合があります
                        if result.min() >= 0 and result.max() <= 1:
                            result = result * 2 - 1
                        
                        # 角度と速度の値を返す
                        return result[0], result[1]
                    
                    def to(self, device):
                        # PyTorchとの互換性のためのダミーメソッド
                        print(f"Note: ONNX models don't support device transfer. Already using {self._device}")
                        return self
                    
                    def eval(self):
                        # PyTorchとの互換性のためのダミーメソッド
                        return self
                
                # ONNXモデルインスタンスを作成
                kl = ONNXModel(ort_session, input_name, preprocess)
                print("ONNX model loaded successfully")
                
            except Exception as e:
                print(f"Error loading ONNX model: {e}")
                import traceback
                traceback.print_exc()
                raise

            # シンプルな画像入力のみ
            inputs = ['cam/image_array']


        ###add pure pytorch model with timm models by romihi@20250407　 
        #pure pytorch models
        elif cfg.DEFAULT_AI_FRAMEWORK == 'pytorch' and '.pth' in model_path:
            import torch
            try:
                from torch2trt import TRTModule
                has_trt = True
            except ImportError:
                has_trt = False
            
            # CUDA利用可能性を確認
            use_cuda = torch.cuda.is_available()
            device = torch.device('cuda' if use_cuda else 'cpu')
            
            from model_catalog import get_model, load_model_weights
            
            # TensorRTモデルパスを設定（元のモデルパスから派生可能に）
            trt_model_path = os.path.splitext(model_path)[0] + '_trt.pth'
            
            # TensorRTモデルが利用可能かチェック
            use_trt = has_trt and use_cuda and os.path.exists(trt_model_path)
            
            try:
                if use_trt:
                    print(f"Loading TensorRT optimized model from {trt_model_path}")
                    kl = TRTModule()
                    kl.load_state_dict(torch.load(trt_model_path))
                    print("TensorRT model loaded successfully")
                else:
                    # TensorRTが使えない理由をログ出力
                    if not has_trt:
                        print("TensorRT not available: package not installed")
                    elif not use_cuda:
                        print("TensorRT not available: CUDA not available")
                    elif not os.path.exists(trt_model_path):
                        print(f"TensorRT model not found at {trt_model_path}")
                    
                    print(f"Loading standard PyTorch model from {model_path}")
                    kl = get_model(model_type, pretrained=False, input_size=(cfg.IMAGE_W, cfg.IMAGE_H))
                    kl = load_model_weights(kl, model_path, device)
                
                # モデルをデバイスに移動
                kl = kl.to(device)
                # 推論モードに設定
                kl.eval()
                    
            except Exception as e:
                print(f"Error loading model: {e}")
                print(f"Attempting to load with alternative method...")
                try:
                    # 代替方法でモデル読み込み試行
                    kl = get_model(model_type, pretrained=False, input_size=(cfg.IMAGE_W, cfg.IMAGE_H))
                    checkpoint = torch.load(model_path, map_location=device)
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        kl.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        kl.load_state_dict(checkpoint)
                    kl = kl.to(device)
                    kl.eval()
                    print("Model loaded with alternative method")
                except Exception as e2:
                    print(f"All loading attempts failed: {e2}")
                    raise

            # シンプルな画像入力のみ
            inputs = ['cam/image_array']

        ###else tub shift add 20230819
        else:
            # If we have a model, create an appropriate Keras part
            kl = dk.utils.get_model_by_type(model_type, cfg)

            #
            # get callback function to reload the model
            # for the configured model format
            #
            
            ###add
            if model_path2 and duo:
                kl2 = dk.utils.get_model_by_type(model_type, cfg)
            ###

            model_reload_cb = None

            
            if '.h5' in model_path or '.trt' in model_path or '.tflite' in \
                model_path or '.savedmodel' in model_path or '.pth' in model_path:
                # load the whole model with weigths, etc
                ###add
                if model_path2:
                    load_model2(kl, model_path,model_path2)

                else:
                    # load the whole model with weigths, etc
                    load_model(kl, model_path)
                ###add
                if model_path2 and duo:
                    #model_path2=model_path2
                    load_model(kl2, model_path2)
                ###

                def reload_model(filename):
                    load_model(kl, filename)

                model_reload_cb = reload_model
        ###

            elif '.json' in model_path:
                # when we have a .json extension
                # load the model from there and look for a matching
                # .wts file with just weights
                load_model_json(kl, model_path)
                weights_path = model_path.replace('.json', '.weights')
                load_weights(kl, weights_path)

                def reload_weights(filename):
                    weights_path = filename.replace('.json', '.weights')
                    load_weights(kl, weights_path)

                model_reload_cb = reload_weights

            else:
                print("ERR>> Unknown extension type on model file!!")
                return

            # this part will signal visual LED, if connected
            V.add(FileWatcher(model_path, verbose=True),
                outputs=['modelfile/modified'])

            # these parts will reload the model file, but only when ai is running
            # so we don't interrupt user driving
            V.add(FileWatcher(model_path), outputs=['modelfile/dirty'],
                run_condition="run_pilot")
            V.add(DelayedTrigger(100), inputs=['modelfile/dirty'],
                outputs=['modelfile/reload'], run_condition="run_pilot")
            V.add(TriggeredCallback(model_path, model_reload_cb),
                inputs=["modelfile/reload"], run_condition="run_pilot")

            #
            # collect inputs to model for inference
            #
            ###
            if cfg.TRAIN_BEHAVIORS:
                bh = BehaviorPart(cfg.BEHAVIOR_LIST)
                V.add(bh, outputs=['behavior/state', 'behavior/label', "behavior/one_hot_state_array"])
                try:
                    ctr.set_button_down_trigger('L1', bh.increment_state)
                except:
                    pass

                inputs = ['cam/image_array', "behavior/one_hot_state_array"]

            elif cfg.USE_LIDAR:
                inputs = ['cam/image_array', 'lidar/dist_array']

            elif cfg.HAVE_ODOM:
                inputs = ['cam/image_array', 'enc/speed']

            elif model_type == "imu":
                assert cfg.HAVE_IMU, 'Missing imu parameter in config'

                class Vectorizer:
                    def run(self, *components):
                        return components

                V.add(Vectorizer, inputs=['imu/acl_x', 'imu/acl_y', 'imu/acl_z',
                                        'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z'],
                    outputs=['imu_array'])

                inputs = ['cam/image_array', 'imu_array']
            ### lap count
            elif cfg.LAP_COUNT and model_path2:
                print("!!!!!!!Lap count switches models!!!!!!!")
                inputs = ['cam/image_array',"lap/one_hot_state_array"]
            ###
            else:
                inputs = ['cam/image_array']

        ### else tub shift

        #
        # collect model inference outputs
        #
        outputs = ['pilot/angle', 'pilot/throttle']
 
        ###add
        if model_path2 and duo:
            outputs2 = ['pilot/angle2', 'pilot/throttle2']
            V.add(kl2, inputs=inputs, outputs=outputs2, run_condition='run_pilot')
 
        ###
        if cfg.TRAIN_LOCALIZER:
            outputs.append("pilot/loc")
        #
        # Add image transformations like crop or trapezoidal mask
        # so they get applied at inference time in autopilot mode.
        #
        if hasattr(cfg, 'TRANSFORMATIONS') or hasattr(cfg, 'POST_TRANSFORMATIONS'):
            from donkeycar.parts.image_transformations import ImageTransformations
            #
            # add the complete set of pre and post augmentation transformations
            #
            logger.info(f"Adding inference transformations")
            V.add(ImageTransformations(cfg, 'TRANSFORMATIONS',
                                       'POST_TRANSFORMATIONS'),
                  inputs=['cam/image_array'], outputs=['cam/image_array_trans'])
            inputs = ['cam/image_array_trans'] + inputs[1:]

        V.add(kl, inputs=inputs, outputs=outputs, run_condition='run_pilot')


    #
    # stop at a stop sign
    #
    if cfg.STOP_SIGN_DETECTOR:
        from donkeycar.parts.object_detector.stop_sign_detector \
            import StopSignDetector
        V.add(StopSignDetector(cfg.STOP_SIGN_MIN_SCORE,
                               cfg.STOP_SIGN_SHOW_BOUNDING_BOX,
                               cfg.STOP_SIGN_MAX_REVERSE_COUNT,
                               cfg.STOP_SIGN_REVERSE_THROTTLE),
              inputs=['cam/image_array', 'pilot/throttle'],
              outputs=['pilot/throttle', 'cam/image_array'])
        V.add(ThrottleFilter(), 
              inputs=['pilot/throttle'],
              outputs=['pilot/throttle'])

    #
    # to give the car a boost when starting ai mode in a race.
    # This will also override the stop sign detector so that
    # you can start at a stop sign using launch mode, but
    # will stop when it comes to the stop sign the next time.
    #
    # NOTE: when launch throttle is in effect, pilot speed is set to None
    #
    aiLauncher = AiLaunch(cfg.AI_LAUNCH_DURATION, cfg.AI_LAUNCH_THROTTLE, cfg.AI_LAUNCH_KEEP_ENABLED)
    V.add(aiLauncher,
          inputs=['user/mode', 'pilot/throttle'],
          outputs=['pilot/throttle'])

    #
    # Decide what inputs should change the car's steering and throttle
    # based on the choice of user or autopilot drive mode
    #
    ###
    '''
    if cfg.HAVE_GS2:
        V.add(DriveMode_GS2(cfg, cfg.AI_THROTTLE_MULT),
            inputs=['user/mode', 'user/angle', 'user/throttle',
                    'pilot/angle', 'pilot/throttle','lidar/dist'],
            outputs=['steering', 'throttle'])
    ''' 
    if cfg.HAVE_TMINI :
        V.add(DriveMode_TMINI(cfg, cfg.AI_THROTTLE_MULT),
            inputs=['user/mode', 'user/angle', 'user/throttle',
                    'pilot/angle', 'pilot/throttle','lidar/dist',
                    'imu/g_thr','imu/g_str'],
            outputs=['steering', 'throttle'])
        
    elif cfg.HAVE_GS2 :
        V.add(DriveMode_GS2(cfg, cfg.AI_THROTTLE_MULT),
            inputs=['user/mode', 'user/angle', 'user/throttle',
                    'pilot/angle', 'pilot/throttle','lidar/dist',
                    'imu/g_thr','imu/g_str'],
            outputs=['steering', 'throttle'])

    #elif model_path2 and duo: 
    #    V.add(LAP_COUNT(cfg.AI_THROTTLE_MULT),
    #      inputs=['user/mode', 'user/angle', 'user/throttle',
    #              'pilot/angle', 'pilot/throttle',
    #              'pilot/angle2', 'pilot/throttle2'],
    #      outputs=['steering', 'throttle'])
    ###
    else: 
        V.add(DriveMode(cfg.AI_THROTTLE_MULT),
          inputs=['user/mode', 'user/angle', 'user/throttle',
                  'pilot/angle', 'pilot/throttle'],
          outputs=['steering', 'throttle'])
        

    if (cfg.CONTROLLER_TYPE != "pigpio_rc") and (cfg.CONTROLLER_TYPE != "MM1"):
        if isinstance(ctr, JoystickController):
            ctr.set_button_down_trigger(cfg.AI_LAUNCH_ENABLE_BUTTON, aiLauncher.enable_ai_launch)


    # Ai Recording
    recording_control = ToggleRecording(cfg.AUTO_RECORD_ON_THROTTLE, cfg.RECORD_DURING_AI)
    V.add(recording_control, inputs=['user/mode', "recording"], outputs=["recording"])

    #
    # Setup drivetrain
    #
    add_drivetrain(V, cfg)


    #
    # OLED display setup
    #
    if cfg.USE_SSD1306_128_32:
        from donkeycar.parts.oled import OLEDPart
        auto_record_on_throttle = cfg.USE_JOYSTICK_AS_DEFAULT and cfg.AUTO_RECORD_ON_THROTTLE
        oled_part = OLEDPart(cfg.SSD1306_128_32_I2C_ROTATION, cfg.SSD1306_RESOLUTION, auto_record_on_throttle)
        V.add(oled_part, inputs=['recording', 'tub/num_records', 'user/mode'], outputs=[], threaded=True)

    #
    # add tub to save data
    #
    if cfg.USE_LIDAR:
        inputs = ['cam/image_array', 'lidar/dist_array', 'user/angle', 'user/throttle', 'user/mode']
        types = ['image_array', 'nparray','float', 'float', 'str']
    else:
        inputs=['cam/image_array','user/angle', 'user/throttle', 'user/mode']
        types=['image_array','float', 'float','str']

    if cfg.HAVE_ODOM:
        inputs += ['enc/speed']
        types += ['float']

    ###
    if cfg.HAVE_OPTICALFLOW_PMW3901:
        inputs += ['vel/x','vel/y']
        types += ['float', 'float']

    if cfg.TRAIN_BEHAVIORS:
        inputs += ['behavior/state', 'behavior/label', "behavior/one_hot_state_array"]
        types += ['int', 'str', 'vector']

    if cfg.CAMERA_TYPE == "D435" and cfg.REALSENSE_D435_DEPTH:
        inputs += ['cam/depth_array']
        types += ['gray16_array']

    if cfg.HAVE_IMU or (cfg.CAMERA_TYPE == "D435" and cfg.REALSENSE_D435_IMU):
        inputs += ['imu/acl_x', 'imu/acl_y', 'imu/acl_z',
            'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z']

        types +=['float', 'float', 'float',
           'float', 'float', 'float']
    ### BNO055
    if cfg.HAVE_IMU_BNO055:
        inputs += ['imu/acl_x', 'imu/acl_y', 'imu/acl_z',
            'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z',
            'imu/angle', 'imu/g_thr','imu/g_str', 'lap/one_hot_state_array']

        types +=['float', 'float', 'float',
           'float', 'float', 'float',
           'float', 'float', 'float', 'list']

    ### TMINI or GS2 lidar
    if cfg.HAVE_TMINI or cfg.HAVE_GS2:
        inputs += ['lidar/dist']
        types += ['list']
        #inputs += ['lidar/dist', 'lidar/wall_info', 'lidar/image_array']
        #types += ['list', 'str', 'image_array']

    ### 
    if cfg.HAVE_COLOR_FILTER:
        inputs += ['filter/one_hot_state_array']
        types += ['vector']

    # rbx
    if cfg.DONKEY_GYM:
        if cfg.SIM_RECORD_LOCATION:
            inputs += ['pos/pos_x', 'pos/pos_y', 'pos/pos_z', 'pos/speed', 'pos/cte']
            types  += ['float', 'float', 'float', 'float', 'float']
        if cfg.SIM_RECORD_GYROACCEL:
            inputs += ['gyro/gyro_x', 'gyro/gyro_y', 'gyro/gyro_z', 'accel/accel_x', 'accel/accel_y', 'accel/accel_z']
            types  += ['float', 'float', 'float', 'float', 'float', 'float']
        if cfg.SIM_RECORD_VELOCITY:
            inputs += ['vel/vel_x', 'vel/vel_y', 'vel/vel_z']
            types  += ['float', 'float', 'float']
        if cfg.SIM_RECORD_LIDAR:
            inputs += ['lidar/dist_array']
            types  += ['nparray']

    if cfg.RECORD_DURING_AI:
        inputs += ['pilot/angle', 'pilot/throttle']
        types += ['float', 'float']

    if cfg.HAVE_PERFMON:
        from donkeycar.parts.perfmon import PerfMonitor
        mon = PerfMonitor(cfg)
        perfmon_outputs = ['perf/cpu', 'perf/mem', 'perf/freq']
        inputs += perfmon_outputs
        types += ['float', 'float', 'float']
        V.add(mon, inputs=[], outputs=perfmon_outputs, threaded=True)

    #
    # Create data storage part
    #
    tub_path = TubHandler(path=cfg.DATA_PATH).create_tub_path() if \
        cfg.AUTO_CREATE_NEW_TUB else cfg.DATA_PATH
    meta += getattr(cfg, 'METADATA', [])
    tub_writer = TubWriter(tub_path, inputs=inputs, types=types, metadata=meta)
    V.add(tub_writer, inputs=inputs, outputs=["tub/num_records"], run_condition='recording')

    # Telemetry (we add the same metrics added to the TubHandler
    if cfg.HAVE_MQTT_TELEMETRY:
        from donkeycar.parts.telemetry import MqttTelemetry
        tel = MqttTelemetry(cfg)
        telem_inputs, _ = tel.add_step_inputs(inputs, types)
        V.add(tel, inputs=telem_inputs, outputs=["tub/queue_size"], threaded=True)

    if cfg.PUB_CAMERA_IMAGES:
        from donkeycar.parts.network import TCPServeValue
        from donkeycar.parts.image import ImgArrToJpg
        pub = TCPServeValue("camera")
        V.add(ImgArrToJpg(), inputs=['cam/image_array'], outputs=['jpg/bin'])
        V.add(pub, inputs=['jpg/bin'])


    if cfg.DONKEY_GYM:
        print("You can now go to http://localhost:%d to drive your car." % cfg.WEB_CONTROL_PORT)
    else:
        print("You can now go to <your hostname.local>:%d to drive your car." % cfg.WEB_CONTROL_PORT)
    if has_input_controller:
        print("You can now move your controller to drive your car.")
        if isinstance(ctr, JoystickController):
            ctr.set_tub(tub_writer.tub)
            ctr.print_controls()

    # run the vehicle
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ, max_loop_count=cfg.MAX_LOOPS)


class ToggleRecording:
    def __init__(self, auto_record_on_throttle, record_in_autopilot):
        """
        Donkeycar Part that manages the recording state.
        """
        self.auto_record_on_throttle = auto_record_on_throttle
        self.record_in_autopilot = record_in_autopilot
        self.recording_latch: bool = None
        self.toggle_latch: bool = False
        self.last_recording = None

    def set_recording(self, recording: bool):
        """
        Set latched recording value to be applied on next call to run()
        :param recording: True to record, False to not record
        """
        self.recording_latch = recording

    def toggle_recording(self):
        """
        Force toggle of recording state on next call to run()
        """
        self.toggle_latch = True

    def run(self, mode: str, recording: bool):
        """
        Set recording based on user/autopilot mode
        :param mode: 'user'|'local_angle'|'local_pilot'
        :param recording: current recording flag
        :return: updated recording flag
        """
        recording_in = recording
        if recording_in != self.last_recording:
            logging.info(f"Recording Change = {recording_in}")

        if self.toggle_latch:
            if self.auto_record_on_throttle:
                logger.info(
                    'auto record on throttle is enabled; ignoring toggle of manual mode.')
            else:
                recording = not self.last_recording
            self.toggle_latch = False

        if self.recording_latch is not None:
            recording = self.recording_latch
            self.recording_latch = None

        if recording and mode != 'user' and not self.record_in_autopilot:
            logging.info("Ignoring recording in auto-pilot mode")
            recording = False

        if self.last_recording != recording:
            logging.info(f"Setting Recording = {recording}")

        self.last_recording = recording

        return recording


class DriveMode:
    def __init__(self,  ai_throttle_mult=1.0):
        """
        :param ai_throttle_mult: scale throttle in autopilot mode
        """
        self.ai_throttle_mult = ai_throttle_mult

    def run(self, mode,
            user_steering, user_throttle,
            pilot_steering, pilot_throttle):
        """
        Main final steering and throttle values based on user mode
        :param mode: 'user'|'local_angle'|'local_pilot'
        :param user_steering: steering value in user (manual) mode
        :param user_throttle: throttle value in user (manual) mode
        :param pilot_steering: steering value in autopilot mode
        :param pilot_throttle: throttle value in autopilot mode
        :return: tuple of (steering, throttle) where throttle is
                 scaled by ai_throttle_mult in autopilot mode
        """
        if mode == 'user':
            return user_steering, user_throttle
        elif mode == 'local_angle':
            return pilot_steering if pilot_steering else 0.0, user_throttle
        return (pilot_steering if pilot_steering else 0.0,
               pilot_throttle * self.ai_throttle_mult if pilot_throttle else 0.0)
###
class DriveMode_2model:
    def __init__(self,  ai_throttle_mult=1.0):
        self.ai_throttle_mult = ai_throttle_mult

    def run(self, mode,
            user_steering, user_throttle,
            pilot_steering, pilot_throttle,
            pilot_steering2, pilot_throttle2):
        if mode == 'user':
            return user_steering, user_throttle
        elif mode == 'local_angle':
            return pilot_steering if pilot_steering else 0.0, user_throttle
        elif mode == 'local_2':
            return pilot_steering2 if pilot_steering2 else 0.0, pilot_throttle2 * self.ai_throttle_mult if pilot_throttle2 else 0.0
        return (pilot_steering if pilot_steering else 0.0,
               pilot_throttle * self.ai_throttle_mult if pilot_throttle else 0.0)

###
class DriveMode_TMINI:
    def __init__(self, cfg, ai_throttle_mult=1.0):
        self.ai_throttle_mult = ai_throttle_mult

        self.dynamic_control = cfg.DYNAMIC_CONTROL

        self.recovery_distance = cfg.RECOVERY_DISTANCE
        self.recovery_detection = 0
        self.recovery_detection_times = cfg.RECOVERY_DETECTION_TIMES
        self.recovery_on = 0
        self.recovery_on_back = 0
        self.recovery_duration = cfg.RECOVERY_DURATION
        self.recovery_duration_back = cfg.RECOVERY_DURATION_BACK
        self.recovery_thr = cfg.RECOVERY_THROTTLE
        self.recovery_str = 0.0
        self.recovery_start = time.time()
        self.recovery_counter = 0
        self.detection = []*cfg.RECOVERY_DETECTION_DIV

    def run(self, mode,
                user_angle, user_throttle,
                pilot_angle, pilot_throttle,
                dist_array, gthr, gstr):
                #dist_array):
        if self.dynamic_control:
            user_throttle = user_throttle * gthr
            user_angle = user_angle * gstr
            if pilot_angle is not None :
                pilot_throttle = pilot_throttle * gthr
                pilot_angle = pilot_angle * gstr
            #print(f"Gthr:{gthr}, Gstr:{gstr}")

        if mode == 'user':
            #print(f'uan:{user_angle}, uth:{user_throttle}')
            return user_angle, user_throttle
        elif mode == 'local_angle':
            return pilot_angle if pilot_angle else 0.0, user_throttle
        ### 
        elif mode == 'local_recovery':
            #self.detection = [i < self.recovery_distance for i in dist_array]
            self.detection = dist_array
            # リカバリーON（前方を検知）
            if 1 in self.detection[:] and not self.recovery_on:
                self.recovery_on = 1
                self.recovery_start = time.time()
            # リカバリーONの時は出力上書き
            if self.recovery_on:
                duration = time.time() - self.recovery_start
                print("*Recovery mode ON*,duration:{:.4f}".format(duration))
                #真ん中で壁→バックあり
                ## 最初の１回目で操作決定
                if 1 in self.detection[1:3] and not self.recovery_on_back: 
                    self.recovery_on_back = 1
                    if self.detection[1] == 0 : #左前方空き
                        self.recovery_str = 1 #1 #0.8
                    elif self.detection[2] == 0 : #右前方空き
                        self.recovery_str = -1 #0.8
                    else: 
                        self.recovery_str = 0 #0.0
                    #self.recovery_str = pilot_angle * -1 #バック時は逆ハンドル
                    #pilot_angle = pilot_angle * -1 #バック時は逆ハンドル
                    pilot_angle = self.recovery_str
                    
                ## ２回目以降は継続
                elif self.recovery_on_back and (duration < self.recovery_duration_back):
                    pilot_throttle = self.recovery_thr
                    #pilot_angle = pilot_angle * -1 #* 0.8 #バック時は逆ハンドル
                    pilot_angle = self.recovery_str #壁にぶつかったときに設定したstr維持
                    print(f"back..., str:{pilot_angle} thr:{pilot_throttle}")
                    pass

                # リカバリー時間内で壁検知（バックなし）
                elif duration <  self.recovery_duration:
                    # リカバリーの種類設定
                    # 左右の壁➔進行方向を変える
                    if 1 in self.detection[0:2]:
                        pilot_angle = 1 #0.8
                    elif 1 in self.detection[2:4]:
                        pilot_angle = -1 #0.8
                else: 
                    print("*Recovery mode OFF")
                    self.recovery_on = 0
                    self.recovery_on_back = 0

            return pilot_angle if pilot_angle else 0.0, \
                pilot_throttle * cfg.AI_THROTTLE_MULT \
                    if pilot_throttle else 0.0

        else:
            return pilot_angle if pilot_angle else 0.0, \
                    pilot_throttle * cfg.AI_THROTTLE_MULT \
                        if pilot_throttle else 0.0

###
class DriveMode_GS2:
    def __init__(self, cfg, ai_throttle_mult=1.0):
        self.ai_throttle_mult = ai_throttle_mult

        self.dynamic_control = cfg.DYNAMIC_CONTROL

        self.recovery_distance = cfg.RECOVERY_DISTANCE
        self.recovery_detection = 0
        self.recovery_detection_times = cfg.RECOVERY_DETECTION_TIMES
        self.recovery_on = 0
        self.recovery_on_back = 0
        self.recovery_duration = cfg.RECOVERY_DURATION
        self.recovery_duration_back = cfg.RECOVERY_DURATION_BACK
        self.recovery_thr = cfg.RECOVERY_THROTTLE
        self.recovery_str = 0.0
        self.recovery_start = time.time()
        self.recovery_counter = 0
        self.detection = []*cfg.RECOVERY_DETECTION_DIV

    def run(self, mode,
                user_angle, user_throttle,
                pilot_angle, pilot_throttle,
                dist_array, gthr, gstr):
                #dist_array):
        if self.dynamic_control:
            user_throttle = user_throttle * gthr
            user_angle = user_angle * gstr
            if pilot_angle is not None :
                pilot_throttle = pilot_throttle * gthr
                pilot_angle = pilot_angle * gstr
            #print(f"Gthr:{gthr}, Gstr:{gstr}")

        if mode == 'user':
            #print(f'uan:{user_angle}, uth:{user_throttle}')
            return user_angle, user_throttle
        elif mode == 'local_angle':
            return pilot_angle if pilot_angle else 0.0, user_throttle
        ### 
        elif mode == 'local_recovery':
            #self.detection = [i < self.recovery_distance for i in dist_array]
            self.detection = dist_array
            # リカバリーON
            if not self.detection == [0,0,0] and not self.recovery_on:
                self.recovery_on = 1
                self.recovery_start = time.time()
            # リカバリーONの時は出力上書き
            if self.recovery_on:
                duration = time.time() - self.recovery_start
                print("*Recovery mode ON*,duration:{:.4f}".format(duration))
                #真ん中で壁→バックあり
                ## 最初の１回目で操作決定
                if self.detection[1] == 1 and not self.recovery_on_back: 
                    self.recovery_on_back = 1
                    if self.detection[0] == 0 and self.detection[2] == 1: #右空き[0,1,1]:
                        self.recovery_str = 0 #-1 #1 #0.8
                    elif self.detection[2] == 0 and self.detection[0] == 1: #左空き[1,1,0]:
                        self.recovery_str = 0 #-1 #0.8
                    else: 
                        self.recovery_str = 0 #-1 #0.0
                    #self.recovery_str = pilot_angle * -1 #バック時は逆ハンドル
                    #pilot_angle = pilot_angle * -1 #バック時は逆ハンドル
                    pilot_angle = self.recovery_str
                    
                ## ２回目以降は継続
                elif self.recovery_on_back and (duration < self.recovery_duration_back):
                    pilot_throttle = self.recovery_thr
                    #pilot_angle = pilot_angle * -1 #* 0.8 #バック時は逆ハンドル
                    pilot_angle = self.recovery_str #壁にぶつかったときに設定したstr維持
                    print(f"back..., str:{pilot_angle} thr:{pilot_throttle}")
                    pass

                # リカバリー時間内で壁検知（バックなし）
                elif duration <  self.recovery_duration:
                    # リカバリーの種類設定
                    # 左右の壁➔進行方向を変える
                    if self.detection[0] == 0: #右空き[0,1,1]:
                        pilot_angle = 1 #0.8
                    elif self.detection[2] == 0: #左空き[1,1,0]:
                        pilot_angle = -1 #0.8
                else: 
                    print("*Recovery mode OFF")
                    self.recovery_on = 0
                    self.recovery_on_back = 0

            return pilot_angle if pilot_angle else 0.0, \
                pilot_throttle * cfg.AI_THROTTLE_MULT \
                    if pilot_throttle else 0.0

        else:
            return pilot_angle if pilot_angle else 0.0, \
                    pilot_throttle * cfg.AI_THROTTLE_MULT \
                        if pilot_throttle else 0.0

### 


class UserPilotCondition:
    def __init__(self, show_pilot_image:bool = False) -> None:
        """
        :param show_pilot_image:bool True to show pilot image in pilot mode
                                     False to show user image in pilot mode
        """
        self.show_pilot_image = show_pilot_image

    def run(self, mode, user_image, pilot_image):
        """
        Maintain run condition and which image to show in web ui
        :param mode: 'user'|'local_angle'|'local_pilot'
        :param user_image: image to show in manual (user) pilot
        :param pilot_image: image to show in auto pilot
        :return: tuple of (user-condition, autopilot-condition, web image)
        """
        if mode == 'user':
            return True, False, user_image
        else:
            return False, True, pilot_image if self.show_pilot_image else user_image


def add_user_controller(V, cfg, use_joystick, input_image='ui/image_array'):
    """
    Add the web controller and any other
    configured user input controller.
    :param V: the vehicle pipeline.
              On output this will be modified.
    :param cfg: the configuration (from myconfig.py)
    :return: the controller
    """

    #
    # This web controller will create a web server that is capable
    # of managing steering, throttle, and modes, and more.
    #
    ctr = LocalWebController(port=cfg.WEB_CONTROL_PORT, mode=cfg.WEB_INIT_MODE)
    V.add(ctr,
          inputs=[input_image, 'tub/num_records', 'user/mode', 'recording'],
          outputs=['user/steering', 'user/throttle', 'user/mode', 'recording', 'web/buttons'],
          threaded=True)

    #
    # also add a physical controller if one is configured
    #
    if use_joystick or cfg.USE_JOYSTICK_AS_DEFAULT:
        #
        # RC controller
        #
        if cfg.CONTROLLER_TYPE == "pigpio_rc":  # an RC controllers read by GPIO pins. They typically don't have buttons
            from donkeycar.parts.controller import RCReceiver
            ctr = RCReceiver(cfg)
            V.add(
                ctr,
                inputs=['user/mode', 'recording'],
                outputs=['user/steering', 'user/throttle',
                         'user/mode', 'recording'],
                threaded=False)
        else:
            #
            # custom game controller mapping created with
            # `donkey createjs` command
            #
            if cfg.CONTROLLER_TYPE == "custom":  # custom controller created with `donkey createjs` command
                from my_joystick import MyJoystickController
                ctr = MyJoystickController(
                    throttle_dir=cfg.JOYSTICK_THROTTLE_DIR,
                    throttle_scale=cfg.JOYSTICK_MAX_THROTTLE,
                    steering_scale=cfg.JOYSTICK_STEERING_SCALE,
                    auto_record_on_throttle=cfg.AUTO_RECORD_ON_THROTTLE)
                ctr.set_deadzone(cfg.JOYSTICK_DEADZONE)
            elif cfg.CONTROLLER_TYPE == "MM1":
                from donkeycar.parts.robohat import RoboHATController
                ctr = RoboHATController(cfg)
            elif cfg.CONTROLLER_TYPE == "mock":
                from donkeycar.parts.controller import MockController
                ctr = MockController(steering=cfg.MOCK_JOYSTICK_STEERING,
                                     throttle=cfg.MOCK_JOYSTICK_THROTTLE)
            else:
                #
                # game controller
                #
                from donkeycar.parts.controller import get_js_controller
                ctr = get_js_controller(cfg)
                if cfg.USE_NETWORKED_JS:
                    from donkeycar.parts.controller import JoyStickSub
                    netwkJs = JoyStickSub(cfg.NETWORK_JS_SERVER_IP)
                    V.add(netwkJs, threaded=True)
                    ctr.js = netwkJs
            V.add(
                ctr,
                inputs=[input_image, 'user/mode', 'recording'],
                outputs=['user/steering', 'user/throttle',
                         'user/mode', 'recording'],
                threaded=True)
    return ctr


def add_simulator(V, cfg):
    # Donkey gym part will output position information if it is configured
    # TODO: the simulation outputs conflict with imu, odometry, kinematics pose estimation and T265 outputs; make them work together.
    if cfg.DONKEY_GYM:
        from donkeycar.parts.dgym import DonkeyGymEnv
        # rbx
        gym = DonkeyGymEnv(cfg.DONKEY_SIM_PATH, host=cfg.SIM_HOST, env_name=cfg.DONKEY_GYM_ENV_NAME, conf=cfg.GYM_CONF,
                           record_location=cfg.SIM_RECORD_LOCATION, record_gyroaccel=cfg.SIM_RECORD_GYROACCEL,
                           record_velocity=cfg.SIM_RECORD_VELOCITY, record_lidar=cfg.SIM_RECORD_LIDAR,
                        #    record_distance=cfg.SIM_RECORD_DISTANCE, record_orientation=cfg.SIM_RECORD_ORIENTATION,
                           delay=cfg.SIM_ARTIFICIAL_LATENCY)
        threaded = True
        inputs = ['steering', 'throttle']
        outputs = ['cam/image_array']

        if cfg.SIM_RECORD_LOCATION:
            outputs += ['pos/pos_x', 'pos/pos_y', 'pos/pos_z', 'pos/speed', 'pos/cte']
        if cfg.SIM_RECORD_GYROACCEL:
            outputs += ['gyro/gyro_x', 'gyro/gyro_y', 'gyro/gyro_z', 'accel/accel_x', 'accel/accel_y', 'accel/accel_z']
        if cfg.SIM_RECORD_VELOCITY:
            outputs += ['vel/vel_x', 'vel/vel_y', 'vel/vel_z']
        if cfg.SIM_RECORD_LIDAR:
            outputs += ['lidar/dist_array']
        # if cfg.SIM_RECORD_DISTANCE:
        #     outputs += ['dist/left', 'dist/right']
        # if cfg.SIM_RECORD_ORIENTATION:
        #     outputs += ['roll', 'pitch', 'yaw']

        V.add(gym, inputs=inputs, outputs=outputs, threaded=threaded)


def get_camera(cfg):
    """
    Get the configured camera part
    """
    cam = None
    if not cfg.DONKEY_GYM:
        if cfg.CAMERA_TYPE == "PICAM":
            from donkeycar.parts.camera import PiCamera
            cam = PiCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH,
                        vflip=cfg.CAMERA_VFLIP, hflip=cfg.CAMERA_HFLIP, camera_index=cfg.CAMERA_INDEX)
            ###               vflip=cfg.CAMERA_VFLIP, hflip=cfg.CAMERA_HFLIP)
        ### add@20250316
        elif cfg.CAMERA_TYPE == "LIDARIMAGE":
            from donkeycar.parts.camera import PiCameraLidar
            cam = PiCameraLidar(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH,
                        vflip=cfg.CAMERA_VFLIP, hflip=cfg.CAMERA_HFLIP, camera_index=cfg.CAMERA_INDEX)

        elif cfg.CAMERA_TYPE == "WEBCAM":
            from donkeycar.parts.camera import Webcam
            cam = Webcam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH)
        elif cfg.CAMERA_TYPE == "CVCAM":
            from donkeycar.parts.cv import CvCam
            cam = CvCam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH)
        elif cfg.CAMERA_TYPE == "CSIC":
            from donkeycar.parts.camera import CSICamera
            cam = CSICamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH,
                            capture_width=cfg.IMAGE_W, capture_height=cfg.IMAGE_H,
                            framerate=cfg.CAMERA_FRAMERATE, gstreamer_flip=cfg.CSIC_CAM_GSTREAMER_FLIP_PARM)
        elif cfg.CAMERA_TYPE == "V4L":
            from donkeycar.parts.camera import V4LCamera
            cam = V4LCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, framerate=cfg.CAMERA_FRAMERATE)
        elif cfg.CAMERA_TYPE == "IMAGE_LIST":
            from donkeycar.parts.camera import ImageListCamera
            cam = ImageListCamera(path_mask=cfg.PATH_MASK)
        elif cfg.CAMERA_TYPE == "LEOPARD":
            from donkeycar.parts.leopard_imaging import LICamera
            cam = LICamera(width=cfg.IMAGE_W, height=cfg.IMAGE_H, fps=cfg.CAMERA_FRAMERATE)
        elif cfg.CAMERA_TYPE == "MOCK":
            from donkeycar.parts.camera import MockCamera
            cam = MockCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH)
        else:
            raise(Exception("Unkown camera type: %s" % cfg.CAMERA_TYPE))
    return cam


def add_camera(V, cfg, camera_type):
    """
    Add the configured camera to the vehicle pipeline.

    :param V: the vehicle pipeline.
              On output this will be modified.
    :param cfg: the configuration (from myconfig.py)
    """
    logger.info("cfg.CAMERA_TYPE %s"%cfg.CAMERA_TYPE)
    if camera_type == "stereo":
        if cfg.CAMERA_TYPE == "WEBCAM":
            from donkeycar.parts.camera import Webcam

            camA = Webcam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, iCam = 0)
            camB = Webcam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, iCam = 1)

        elif cfg.CAMERA_TYPE == "CVCAM":
            from donkeycar.parts.cv import CvCam

            camA = CvCam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, iCam = 0)
            camB = CvCam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, iCam = 1)
        else:
            raise(Exception("Unsupported camera type: %s" % cfg.CAMERA_TYPE))

        V.add(camA, outputs=['cam/image_array_a'], threaded=True)
        V.add(camB, outputs=['cam/image_array_b'], threaded=True)

        from donkeycar.parts.image import StereoPair

        V.add(StereoPair(), inputs=['cam/image_array_a', 'cam/image_array_b'],
            outputs=['cam/image_array'])
        if cfg.BGR2RGB:
            from donkeycar.parts.cv import ImgBGR2RGB
            V.add(ImgBGR2RGB(), inputs=["cam/image_array_a"], outputs=["cam/image_array_a"])
            V.add(ImgBGR2RGB(), inputs=["cam/image_array_b"], outputs=["cam/image_array_b"])

    elif cfg.CAMERA_TYPE == "D435":
        from donkeycar.parts.realsense435i import RealSense435i
        cam = RealSense435i(
            enable_rgb=cfg.REALSENSE_D435_RGB,
            enable_depth=cfg.REALSENSE_D435_DEPTH,
            enable_imu=cfg.REALSENSE_D435_IMU,
            device_id=cfg.REALSENSE_D435_ID)
        V.add(cam, inputs=[],
              outputs=['cam/image_array', 'cam/depth_array',
                       'imu/acl_x', 'imu/acl_y', 'imu/acl_z',
                       'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z'],
              threaded=True)
    ### for lidar imaging
    elif cfg.CAMERA_TYPE == "LIDARIMAGE":
        inputs = ['lidar/image_array']
        outputs = ['cam/image_array']
        threaded = True
        cam = get_camera(cfg)
        V.add(cam, inputs=inputs, outputs=outputs, threaded=threaded)

    else:
        inputs = []
        outputs = ['cam/image_array']
        threaded = True
        cam = get_camera(cfg)
        if cam:
            V.add(cam, inputs=inputs, outputs=outputs, threaded=threaded)
        if cfg.BGR2RGB:
            from donkeycar.parts.cv import ImgBGR2RGB
            V.add(ImgBGR2RGB(), inputs=["cam/image_array"], outputs=["cam/image_array"])


def add_odometry(V, cfg, threaded=True):
    """
    If the configuration support odometry, then
    add encoders, odometry and kinematics to the vehicle pipeline
    :param V: the vehicle pipeline.
              On output this may be modified.
    :param cfg: the configuration (from myconfig.py)
    """
    from donkeycar.parts.pose import BicyclePose, UnicyclePose

    if cfg.HAVE_ODOM:
        poll_delay_secs = 0.01  # pose estimation runs at 100hz
        kinematics = UnicyclePose(cfg, poll_delay_secs) if cfg.HAVE_ODOM_2 else BicyclePose(cfg, poll_delay_secs)
        V.add(kinematics,
            inputs = ["throttle", "steering", None],
            outputs = ['enc/distance', 'enc/speed', 'pos/x', 'pos/y',
                       'pos/angle', 'vel/x', 'vel/y', 'vel/angle',
                       'nul/timestamp'],
            threaded = threaded)

###
def add_opticalflow(V, cfg, threaded=True):
    if cfg.HAVE_OPTICALFLOW_PMW3901:
        import opticalflow_pmw3901
        #poll_delay_secs = 0.01  # pose estimation runs at 100hz
        opticalflow = opticalflow_pmw3901.OPTICALFLOW()
        V.add(opticalflow,
            inputs = [],
            outputs = ['vel/x', 'vel/y'],
            threaded = threaded)

#
# IMU setup
#
def add_imu(V, cfg):
    imu = None
    if cfg.HAVE_IMU:
        from donkeycar.parts.imu import IMU

        imu = IMU(sensor=cfg.IMU_SENSOR, addr=cfg.IMU_ADDRESS,
                  dlp_setting=cfg.IMU_DLP_CONFIG)
        V.add(imu, outputs=['imu/acl_x', 'imu/acl_y', 'imu/acl_z',
                            'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z'], threaded=True)
    ###
    elif cfg.HAVE_IMU_BNO055:
        import  gyro
        imu = gyro.BNO055(cfg.G_MODE)
        imu.Gmode = cfg.G_MODE
        V.add(imu, outputs=['imu/acl_x', 'imu/acl_y', 'imu/acl_z',
                            'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z', 
                            'imu/angle', 'imu/g_thr','imu/g_str','lap/one_hot_state_array'], threaded=False)
 
    return imu


#
# Drive train setup
#
def add_drivetrain(V, cfg):

    if (not cfg.DONKEY_GYM) and cfg.DRIVE_TRAIN_TYPE != "MOCK":
        from donkeycar.parts import actuator, pins
        from donkeycar.parts.actuator import TwoWheelSteeringThrottle

        #
        # To make differential drive steer,
        # divide throttle between motors based on the steering value
        #
        is_differential_drive = cfg.DRIVE_TRAIN_TYPE.startswith("DC_TWO_WHEEL")
        if is_differential_drive:
            V.add(TwoWheelSteeringThrottle(),
                  inputs=['throttle', 'steering'],
                  outputs=['left/throttle', 'right/throttle'])

        if cfg.DRIVE_TRAIN_TYPE == "PWM_STEERING_THROTTLE":
            #
            # drivetrain for RC car with servo and ESC.
            # using a PwmPin for steering (servo)
            # and as second PwmPin for throttle (ESC)
            #
            from donkeycar.parts.actuator import PWMSteering, PWMThrottle, PulseController

            dt = cfg.PWM_STEERING_THROTTLE
            steering_controller = PulseController(
                pwm_pin=pins.pwm_pin_by_id(dt["PWM_STEERING_PIN"]),
                pwm_scale=dt["PWM_STEERING_SCALE"],
                pwm_inverted=dt["PWM_STEERING_INVERTED"])
            steering = PWMSteering(controller=steering_controller,
                                            left_pulse=dt["STEERING_LEFT_PWM"],
                                            right_pulse=dt["STEERING_RIGHT_PWM"])

            throttle_controller = PulseController(
                pwm_pin=pins.pwm_pin_by_id(dt["PWM_THROTTLE_PIN"]),
                pwm_scale=dt["PWM_THROTTLE_SCALE"],
                pwm_inverted=dt['PWM_THROTTLE_INVERTED'])
            throttle = PWMThrottle(controller=throttle_controller,
                                                max_pulse=dt['THROTTLE_FORWARD_PWM'],
                                                zero_pulse=dt['THROTTLE_STOPPED_PWM'],
                                                min_pulse=dt['THROTTLE_REVERSE_PWM'])
            V.add(steering, inputs=['steering'], threaded=True)
            V.add(throttle, inputs=['throttle'], threaded=True)

        elif cfg.DRIVE_TRAIN_TYPE == "I2C_SERVO":
            #
            # This driver is DEPRECATED in favor of 'DRIVE_TRAIN_TYPE == "PWM_STEERING_THROTTLE"'
            # This driver will be removed in a future release
            #
            from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle

            steering_controller = PCA9685(cfg.STEERING_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
            steering = PWMSteering(controller=steering_controller,
                                            left_pulse=cfg.STEERING_LEFT_PWM,
                                            right_pulse=cfg.STEERING_RIGHT_PWM)

            throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
            throttle = PWMThrottle(controller=throttle_controller,
                                            max_pulse=cfg.THROTTLE_FORWARD_PWM,
                                            zero_pulse=cfg.THROTTLE_STOPPED_PWM,
                                            min_pulse=cfg.THROTTLE_REVERSE_PWM)

            V.add(steering, inputs=['steering'], threaded=True)
            V.add(throttle, inputs=['throttle'], threaded=True)

        elif cfg.DRIVE_TRAIN_TYPE == "DC_STEER_THROTTLE":
            dt = cfg.DC_STEER_THROTTLE
            steering = actuator.L298N_HBridge_2pin(
                pins.pwm_pin_by_id(dt['LEFT_DUTY_PIN']),
                pins.pwm_pin_by_id(dt['RIGHT_DUTY_PIN']))
            throttle = actuator.L298N_HBridge_2pin(
                pins.pwm_pin_by_id(dt['FWD_DUTY_PIN']),
                pins.pwm_pin_by_id(dt['BWD_DUTY_PIN']))

            V.add(steering, inputs=['steering'])
            V.add(throttle, inputs=['throttle'])

        elif cfg.DRIVE_TRAIN_TYPE == "DC_TWO_WHEEL":
            dt = cfg.DC_TWO_WHEEL
            left_motor = actuator.L298N_HBridge_2pin(
                pins.pwm_pin_by_id(dt['LEFT_FWD_DUTY_PIN']),
                pins.pwm_pin_by_id(dt['LEFT_BWD_DUTY_PIN']))
            right_motor = actuator.L298N_HBridge_2pin(
                pins.pwm_pin_by_id(dt['RIGHT_FWD_DUTY_PIN']),
                pins.pwm_pin_by_id(dt['RIGHT_BWD_DUTY_PIN']))

            V.add(left_motor, inputs=['left/throttle'])
            V.add(right_motor, inputs=['right/throttle'])

        elif cfg.DRIVE_TRAIN_TYPE == "DC_TWO_WHEEL_L298N":
            dt = cfg.DC_TWO_WHEEL_L298N
            left_motor = actuator.L298N_HBridge_3pin(
                pins.output_pin_by_id(dt['LEFT_FWD_PIN']),
                pins.output_pin_by_id(dt['LEFT_BWD_PIN']),
                pins.pwm_pin_by_id(dt['LEFT_EN_DUTY_PIN']))
            right_motor = actuator.L298N_HBridge_3pin(
                pins.output_pin_by_id(dt['RIGHT_FWD_PIN']),
                pins.output_pin_by_id(dt['RIGHT_BWD_PIN']),
                pins.pwm_pin_by_id(dt['RIGHT_EN_DUTY_PIN']))

            V.add(left_motor, inputs=['left/throttle'])
            V.add(right_motor, inputs=['right/throttle'])

        elif cfg.DRIVE_TRAIN_TYPE == "SERVO_HBRIDGE_2PIN":
            #
            # Servo for steering and HBridge motor driver in 2pin mode for motor
            #
            from donkeycar.parts.actuator import PWMSteering, PWMThrottle, PulseController

            dt = cfg.SERVO_HBRIDGE_2PIN
            steering_controller = PulseController(
                pwm_pin=pins.pwm_pin_by_id(dt['PWM_STEERING_PIN']),
                pwm_scale=dt['PWM_STEERING_SCALE'],
                pwm_inverted=dt['PWM_STEERING_INVERTED'])
            steering = PWMSteering(controller=steering_controller,
                                            left_pulse=dt['STEERING_LEFT_PWM'],
                                            right_pulse=dt['STEERING_RIGHT_PWM'])

            motor = actuator.L298N_HBridge_2pin(
                pins.pwm_pin_by_id(dt['FWD_DUTY_PIN']),
                pins.pwm_pin_by_id(dt['BWD_DUTY_PIN']))

            V.add(steering, inputs=['steering'], threaded=True)
            V.add(motor, inputs=["throttle"])

        elif cfg.DRIVE_TRAIN_TYPE == "SERVO_HBRIDGE_3PIN":
            #
            # Servo for steering and HBridge motor driver in 3pin mode for motor
            #
            from donkeycar.parts.actuator import PWMSteering, PWMThrottle, PulseController

            dt = cfg.SERVO_HBRIDGE_3PIN
            steering_controller = PulseController(
                pwm_pin=pins.pwm_pin_by_id(dt['PWM_STEERING_PIN']),
                pwm_scale=dt['PWM_STEERING_SCALE'],
                pwm_inverted=dt['PWM_STEERING_INVERTED'])
            steering = PWMSteering(controller=steering_controller,
                                            left_pulse=dt['STEERING_LEFT_PWM'],
                                            right_pulse=dt['STEERING_RIGHT_PWM'])

            motor = actuator.L298N_HBridge_3pin(
                pins.output_pin_by_id(dt['FWD_PIN']),
                pins.output_pin_by_id(dt['BWD_PIN']),
                pins.pwm_pin_by_id(dt['DUTY_PIN']))

            V.add(steering, inputs=['steering'], threaded=True)
            V.add(motor, inputs=["throttle"])

        elif cfg.DRIVE_TRAIN_TYPE == "SERVO_HBRIDGE_PWM":
            #
            # This driver is DEPRECATED in favor of 'DRIVE_TRAIN_TYPE == "SERVO_HBRIDGE_2PIN"'
            # This driver will be removed in a future release
            #
            from donkeycar.parts.actuator import ServoBlaster, PWMSteering
            steering_controller = ServoBlaster(cfg.STEERING_CHANNEL) #really pin
            # PWM pulse values should be in the range of 100 to 200
            assert(cfg.STEERING_LEFT_PWM <= 200)
            assert(cfg.STEERING_RIGHT_PWM <= 200)
            steering = PWMSteering(controller=steering_controller,
                                   left_pulse=cfg.STEERING_LEFT_PWM,
                                   right_pulse=cfg.STEERING_RIGHT_PWM)

            from donkeycar.parts.actuator import Mini_HBridge_DC_Motor_PWM
            motor = Mini_HBridge_DC_Motor_PWM(cfg.HBRIDGE_PIN_FWD, cfg.HBRIDGE_PIN_BWD)

            V.add(steering, inputs=['steering'], threaded=True)
            V.add(motor, inputs=["throttle"])

        elif cfg.DRIVE_TRAIN_TYPE == "MM1":
            from donkeycar.parts.robohat import RoboHATDriver
            V.add(RoboHATDriver(cfg), inputs=['steering', 'throttle'])

        elif cfg.DRIVE_TRAIN_TYPE == "PIGPIO_PWM":
            #
            # This driver is DEPRECATED in favor of 'DRIVE_TRAIN_TYPE == "PWM_STEERING_THROTTLE"'
            # This driver will be removed in a future release
            #
            from donkeycar.parts.actuator import PWMSteering, PWMThrottle, PiGPIO_PWM
            steering_controller = PiGPIO_PWM(cfg.STEERING_PWM_PIN, freq=cfg.STEERING_PWM_FREQ,
                                             inverted=cfg.STEERING_PWM_INVERTED)
            steering = PWMSteering(controller=steering_controller,
                                   left_pulse=cfg.STEERING_LEFT_PWM,
                                   right_pulse=cfg.STEERING_RIGHT_PWM)

            throttle_controller = PiGPIO_PWM(cfg.THROTTLE_PWM_PIN, freq=cfg.THROTTLE_PWM_FREQ,
                                             inverted=cfg.THROTTLE_PWM_INVERTED)
            throttle = PWMThrottle(controller=throttle_controller,
                                   max_pulse=cfg.THROTTLE_FORWARD_PWM,
                                   zero_pulse=cfg.THROTTLE_STOPPED_PWM,
                                   min_pulse=cfg.THROTTLE_REVERSE_PWM)
            V.add(steering, inputs=['steering'], threaded=True)
            V.add(throttle, inputs=['throttle'], threaded=True)
    
        elif cfg.DRIVE_TRAIN_TYPE == "VESC":
            from donkeycar.parts.actuator import VESC
            logger.info("Creating VESC at port {}".format(cfg.VESC_SERIAL_PORT))
            vesc = VESC(cfg.VESC_SERIAL_PORT,
                          cfg.VESC_MAX_SPEED_PERCENT,
                          cfg.VESC_HAS_SENSOR,
                          cfg.VESC_START_HEARTBEAT,
                          cfg.VESC_BAUDRATE,
                          cfg.VESC_TIMEOUT,
                          cfg.VESC_STEERING_SCALE,
                          cfg.VESC_STEERING_OFFSET
                        )
            V.add(vesc, inputs=['steering', 'throttle'])


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config(myconfig=args['--myconfig'])

    if args['drive']:
        model_type = args['--type']
        camera_type = args['--camera']
        drive(cfg, model_path=args['--model'], use_joystick=args['--js'],
              ###
              model_path2=args['--model2'],
              #model_path2=args['--model2'],duo=args['--duo'],
              ###
              model_type=model_type, camera_type=camera_type,
              meta=args['--meta'])

    elif args['train']:
        print('Use python train.py instead.\n')
