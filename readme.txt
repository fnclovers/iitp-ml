## 머신러닝 사용법
두 가지 모델이 구현되었습니다.
1. 파이토치를 이용한 모델
pytorch/mlp_model.py를 실행하여 학습을 진행합니다.
학습 결과는 다음과 같은 포멧으로 저장됩니다.
```
8바이트의 unsigned long long len // 맨 처음에는 학습에 사용된 데이터의 수가 저장됩니다.
<len> 만큼 반복
    8바이트의 unsigned long long LPN // LPN
    8바이트의 unsigned long long  interval // interval
```
2. c++를 이용한 모델
파이토치 c++ api를 이용하여 학습을 진행합니다.
실시간으로 학습과 추론을 진행하게 됩니다.

## 시뮬레이터와 결합하는 법
```cpp
void PageLevelFTL::poll_prediction(uint64_t t, uint64_t d) {
  UNUSED(t);
  Request *cmd = (Request *)d;
  ML::CoReadPrediction prediction;

  if (object.predictor->tryGetPrediction(cmd->getMLTag(), prediction)) {
    ftlobject.pJobManager->triggerByUser(TriggerType::ReadMapping, cmd);
    ftlobject.pMapping->readMapping(cmd, eventReadSubmit);
  }
  else {
    // Prediction not ready, so try again
    scheduleRel(eventPollPrediction, (uint64_t)cmd, 1000000);
  }
}

void PageLevelFTL::read(Request *cmd) {
  uint64_t t = object.predictor->startPrediction(cmd->getLPN());
  cmd->setMLTag(t);
  scheduleNow(eventPollPrediction, (uint64_t)cmd);
}
```
여기에, 읽기를 수행할 때 추론을 수행하는 예시 코드가 있습니다.
학습 지연시간은 TrainingLatency를, 추론 지연시간은 ModelLatency를 수정하여 조절할 수 있습니다.
