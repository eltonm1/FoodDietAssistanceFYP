//
//  ChatBotMessage.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 14/3/2023.
//

import Foundation

struct ChatBotMessage: IdentifyEquateCodeHashable {
    let id = UUID()
    let productEntity: String
    let clientInput: String
    var intentType: IntentType? = .undefined
    var response: String?
    var productsResponse: [Product] = []
    
    enum CodingKeys: String, CodingKey {
        case productEntity = "prod_entity"
        case clientInput = "client_input"
        case intentType = "intent_id"
        case response = "response"
        case productsResponse = "products_response"
    }
    
    var isBot: Bool {
        response != nil
    }
}
