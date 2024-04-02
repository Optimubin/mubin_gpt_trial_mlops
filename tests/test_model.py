from model.model import train_model

def test_model_training():
    model = train_model()
    assert model is not None

def test_model_prediction():
    model = train_model()
    sample_input = [[5.1, 3.5, 1.4, 0.2]]
    prediction = model.predict(sample_input)
    assert prediction is not None
