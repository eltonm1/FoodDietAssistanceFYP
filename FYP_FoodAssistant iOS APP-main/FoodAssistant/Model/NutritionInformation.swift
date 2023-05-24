//
//  NutritionInformation.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 6/12/2022.
//

import Foundation

/// Note: All units are per 100ml, not per serving

struct NutritionInformation: IdentifyEquateCodeHashable {
    let id: Int
    var energy: Int = 0
    var protein: Double = 0
    var total_fat: Double = 0
    var saturated_fat: Double = 0
    var trans_fat: Double = 0
    var carbohydrates: Double = 0
    var sugars: Double = 0
    var sodium: Double = 0
    var cholesterol: Double?
    var vitaminB2: Double?
    var vitaminB3: Double?
    var vitaminB6: Double?
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(self.id, forKey: .id)
        try container.encode(self.energy, forKey: .energy)
        try container.encode(self.protein, forKey: .protein)
        try container.encode(self.total_fat, forKey: .total_fat)
        try container.encode(self.saturated_fat, forKey: .saturated_fat)
        try container.encode(self.trans_fat, forKey: .trans_fat)
        try container.encode(self.carbohydrates, forKey: .carbohydrates)
        try container.encode(self.sugars, forKey: .sugars)
        try container.encode(self.sodium, forKey: .sodium)
        try container.encodeIfPresent(self.cholesterol, forKey: .cholesterol)
        try container.encodeIfPresent(self.vitaminB2, forKey: .vitaminB2)
        try container.encodeIfPresent(self.vitaminB3, forKey: .vitaminB3)
        try container.encodeIfPresent(self.vitaminB6, forKey: .vitaminB6)
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        if container.contains(.id) {
            self.id = try container.decodeIfPresent(Int.self, forKey: .id) ?? UUID().hashValue
        } else {
            self.id = UUID().hashValue
        }
        self.energy = try container.decode(Int.self, forKey: .energy)
        self.protein = try container.decode(Double.self, forKey: .protein)
        self.total_fat = try container.decode(Double.self, forKey: .total_fat)
        self.saturated_fat = try container.decode(Double.self, forKey: .saturated_fat)
        self.trans_fat = try container.decode(Double.self, forKey: .trans_fat)
        self.carbohydrates = try container.decode(Double.self, forKey: .carbohydrates)
        self.sugars = try container.decode(Double.self, forKey: .sugars)
        self.sodium = try container.decode(Double.self, forKey: .sodium)
        self.cholesterol = try container.decodeIfPresent(Double.self, forKey: .cholesterol)
        self.vitaminB2 = try container.decodeIfPresent(Double.self, forKey: .vitaminB2)
        self.vitaminB3 = try container.decodeIfPresent(Double.self, forKey: .vitaminB3)
        self.vitaminB6 = try container.decodeIfPresent(Double.self, forKey: .vitaminB6)
    }
    
    init(
        id: Int,
        energy: Int,
        protein: Double,
        total_fat: Double,
        saturated_fat: Double,
        trans_fat: Double,
        carbohydrates: Double,
        sugars: Double,
        sodium: Double,
        cholesterol: Double,
        vitaminB2: Double?,
        vitaminB3: Double?,
        vitaminB6: Double?
    ) {
        self.id = id
        self.energy = energy
        self.protein = protein
        self.total_fat = total_fat
        self.saturated_fat = saturated_fat
        self.trans_fat = trans_fat
        self.carbohydrates = carbohydrates
        self.sugars = sugars
        self.sodium = sodium
        self.cholesterol = cholesterol
        self.vitaminB2 = vitaminB2
        self.vitaminB3 = vitaminB3
        self.vitaminB6 = vitaminB6
    }
    
    enum CodingKeys: String, CodingKey {
        case id = "id"
        case energy = "energy"
        case protein = "protein"
        case total_fat = "total_fat"
        case saturated_fat = "saturated_fat"
        case trans_fat = "trans_fat"
        case carbohydrates = "carbohydrates"
        case sugars = "sugars"
        case sodium = "sodium"
        case cholesterol = "cholesterol"
        case vitaminB2 = "vitaminB2"
        case vitaminB3 = "vitaminB3"
        case vitaminB6 = "vitaminB6"
    }
}
