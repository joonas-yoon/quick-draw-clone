import torch
import torch.nn as nn
import torch.onnx as onnx


def export_to_onnx(model: nn.Module, filepath: str, input_shape: tuple):
    onnx.export(model,                     # 실행될 모델
                torch.randn(input_shape),  # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                filepath,                  # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                opset_version=13,          # 모델을 변환할 때 사용할 ONNX 버전
                do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                input_names=['input'],     # 모델의 입력값을 가리키는 이름
                output_names=['output'],   # 모델의 출력값을 가리키는 이름
                dynamic_axes={             # 가변적인 길이를 가진 차원
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                })
