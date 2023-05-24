//
//  IntentType.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 14/3/2023.
//

enum IntentType: Int, Codable {
    case greeting = 0
    case whatCanYouDo = 1
    case productPrice = 2
    case productDetails = 3
    case whereToBuyProduct = 4
    case findSimilarProduct = 5
    case undefined = 6
}
