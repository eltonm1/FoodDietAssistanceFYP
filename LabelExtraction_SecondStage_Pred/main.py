from pred_secondstage.pred import NutritionLabelInformationExtractor

extractor = NutritionLabelInformationExtractor()

for pic in ([0, 24, 21, 30, 41, 49]):
    print(pic)
    ket_value = extractor.predict(f"real_uncropped/{pic}.jpeg")
    print(ket_value)
    print("\n\n\n")