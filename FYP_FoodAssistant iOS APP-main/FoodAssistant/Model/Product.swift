//
//  Product.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 20/11/2022.
//

import Foundation
import SwiftUI
import Charts

struct Product: IdentifyEquateCodeHashable {
    
    let id: String
    /// Note:
    /// Because in the `InputFoodProductDetailView`, user can modify the product's name
    /// therefore we are declaring it to be `var` instead of `let` to let `name` be modifiable
    var name: String
    let barcode: String
    var nutrition: NutritionInformation?
    let manufacturer: String?
    let brand: String?
    let prices: [ProductPrice]
    let category1: String?
    let category2: String?
    let category3: String?
//    var ingredients: [Ingredient] = []
    //var supermarket: Supermarket?
    let photo: Photo?
    
    init(
        id: String = UUID().description,
        name: String = "",
        barcode: String = "",
        nutrition: NutritionInformation? = nil,
        manufacturer: String? = nil,
        brand: String? = nil,
        prices: [ProductPrice] = [],
        category1: String? = nil,
        category2: String? = nil,
        category3: String? = nil,
        photo: Photo? = nil
    ) {
        self.id = id
        self.name = name
        self.barcode = barcode
        self.nutrition = nutrition
        self.manufacturer = manufacturer
        self.brand = brand
        self.prices = prices
        self.category1 = category1
        self.category2 = category2
        self.category3 = category3
        self.photo = photo
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        name = try container.decode(String.self, forKey: .name)
        barcode = try container.decode(String.self, forKey: .barcode)
        if container.contains(.nutrition){
            nutrition = try container.decodeIfPresent(NutritionInformation.self, forKey: .nutrition)
        }else{
            nutrition = nil
        }
        
        manufacturer = try container.decode(String.self, forKey: .manufacturer)
        brand = try container.decode(String.self, forKey: .brand)
        prices = try container.decode([ProductPrice].self, forKey: .prices)
        category1 = try container.decode(String.self, forKey: .category1)
        category2 = try container.decode(String.self, forKey: .category2)
        category3 = try container.decode(String.self, forKey: .category3)
        photo = nil
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(id, forKey: .id)
        try container.encodeIfPresent(name, forKey: .name)
        try container.encodeIfPresent(barcode, forKey: .barcode)
        try container.encodeIfPresent(nutrition, forKey: .nutrition)
        try container.encodeIfPresent(manufacturer, forKey: .manufacturer)
        try container.encodeIfPresent(brand, forKey: .brand)
        try container.encodeIfPresent(prices, forKey: .prices)
        try container.encodeIfPresent(category1, forKey: .category1)
        try container.encodeIfPresent(category2, forKey: .category2)
        try container.encodeIfPresent(category3, forKey: .category3)
    }
    
    enum CodingKeys: String, CodingKey {
        case id = "id"
        case name = "name"
        case barcode = "barcode"
        case nutrition = "nutrition"
        case manufacturer = "manufacturer"
        case brand = "brand"
        case prices = "product_price"
        case category1 = "category_1"
        case category2 = "category_2"
        case category3 = "category_3"
        case photo = "photo"
    }
}

struct ProductPrice: IdentifyEquateCodeHashable {
    let id = UUID()
    let price: Double
    let supermarket: Supermarket
    let date: Date
}
