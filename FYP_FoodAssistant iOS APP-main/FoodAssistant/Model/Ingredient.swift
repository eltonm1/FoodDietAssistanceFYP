//
//  Ingredient.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 6/12/2022.
//

import Foundation

enum Ingredient: Codable, CaseIterable, Hashable, Comparable {
    static var allCases: [Ingredient] {
        [
            .purifiedWater,
            .fullCreamEvaporatedMilk,
            .sugar,
            .milkPowder,
            .solubleCoffee,
            .cream,
            .emulsifier(code: ""),
            .regulator(code: ""),
            .carbonatedWater,
            .sucrose,
            .glucose,
            .taurine,
            .flavourings,
            .caffeine,
            .vitamins(code: ""),
            .colours(code: ""),
            .salt,
            .antioxidant(code: ""),
            .barley,
            .oolongTea,
            .greenTea,
            .ltheanine,
            .fructose,
            .honey,
            .peach,
            .soyabean,
            .milk,
            .nut,
            .wheat,
            .egg,
            .peanut,
            .sesame
        ]
    }
    
    static func >(lhs: Ingredient, rhs: Ingredient) -> Bool {
        lhs.englishName > rhs.englishName
    }
    
    case purifiedWater
    case fullCreamEvaporatedMilk
    case sugar
    case milkPowder
    case solubleCoffee
    case cream
    case emulsifier(code: String)
    case regulator(code: String)
    case carbonatedWater
    case sucrose
    case glucose
    case taurine
    case flavourings
    case caffeine
    case vitamins(code: String)
    case colours(code: String)
    case salt
    case antioxidant(code: String)
    case barley
    case oolongTea
    case greenTea
    case ltheanine
    case fructose
    case honey
    case peach
    case soyabean
    case milk
    case nut
    case wheat
    case egg
    case peanut
    case sesame

    var englishName: String {
        switch self {
        case .purifiedWater:
            return "Purified Water"
        case .fullCreamEvaporatedMilk:
            return "Full Cream Evaporated Milk"
        case .sugar:
            return "Sugar"
        case .milkPowder:
            return "Milk Powder"
        case .solubleCoffee:
            return "Soluble Coffee"
        case .cream:
            return "Cream"
        case .emulsifier(let code):
            return "Emulsifier \(code)"
        case .regulator(let code):
            return "Regulator \(code)"
        case .carbonatedWater:
            return "Carbonated Water"
        case .sucrose:
            return "Sucrose"
        case .glucose:
            return "Glucose"
        case .taurine:
            return "Taurine"
        case .flavourings:
            return "Flavourings"
        case .caffeine:
            return "Caffeine"
        case .vitamins(let code):
            return "Vitamins \(code)"
        case .colours(let code):
            return "Colours \(code)"
        case .salt:
            return "Salt"
        case .antioxidant(let code):
            return "Antioxidant \(code)"
        case .barley:
            return "Barley"
        case .oolongTea:
            return "Oolong Tea"
        case .greenTea:
            return "Green Tea"
        case .ltheanine:
            return "L-Theanine"
        case .fructose:
            return "Fructose"
        case .honey:
            return "Honey"
        case .peach:
            return "Peach"
        case .soyabean:
            return "Soyabean"
        case .milk:
            return "Milk"
        case .nut:
            return "Nut"
        case .wheat:
            return "Wheat"
        case .egg:
            return "Egg"
        case .peanut:
            return "Peanut"
        case .sesame:
            return "Sesame"
        }
    }

    var chineseName: String {
        switch self {
        case .purifiedWater:
            return "纯水"
        case .fullCreamEvaporatedMilk:
            return "全脂淡奶"
        case .sugar:
            return "糖"
        case .milkPowder:
            return "奶粉"
        case .solubleCoffee:
            return "即溶咖啡"
        case .cream:
            return "忌廉"
        case .emulsifier(let code):
            return "乳化劑 \(code)"
        case .regulator(let code):
            return "酸度調節劑 \(code)"
        case .carbonatedWater:
            return "碳酸水"
        case .sucrose:
            return "蔗糖"
        case .glucose:
            return "葡萄糖"
        case .taurine:
            return "牛磺酸/氨基乙磺酸"
        case .flavourings:
            return "調味劑"
        case .caffeine:
            return "咖啡因"
        case .vitamins(let code):
            return "維他命 \(code)"
        case .colours(let code):
            return "色素 \(code)"
        case .salt:
            return "鹽"
        case .antioxidant(let code):
            return "抗氧化劑 \(code)"
        case .barley:
            return "大麥"
        case .oolongTea:
            return "烏龍茶"
        case .greenTea:
            return "綠茶"
        case .ltheanine:
            return "茶氨酸"
        case .fructose:
            return "果糖"
        case .honey:
            return "蜂蜜"
        case .peach:
            return "水蜜桃"
        case .soyabean:
            return "大豆"
        case .milk:
            return "牛奶"
        case .nut:
            return "堅果"
        case .wheat:
            return "小麥"
        case .egg:
            return "蛋"
        case .peanut:
            return "花生"
        case .sesame:
            return "芝麻"
        }
    }
    
}
