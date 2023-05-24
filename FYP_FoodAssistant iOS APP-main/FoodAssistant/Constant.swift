//
//  Constant.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 19/3/2023.
//

import SwiftDate
import Foundation

let dummyProducts: [Product] = [
    Product(
        name: "Mandarin Lemon Juice Drink 500mL",
        barcode: "4892214250168",
        nutrition: nil,
        manufacturer: nil,
        brand: "Tao Ti",
        prices: [
            ProductPrice(
                price: 8.9,
                supermarket: .aeon,
                date: "2023-03-16T11:23:13Z".toISODate()!.date
            ),
            ProductPrice(
                price: 28.9,
                supermarket: .aeon,
                date: "2023-03-19T11:23:13Z".toISODate()!.date
            ),
            ProductPrice(
                price: 8.9,
                supermarket: .aeon,
                date: "2023-03-15T11:23:13Z".toISODate()!.date
            ),
            ProductPrice(
                price: 28.9,
                supermarket: .aeon,
                date: "2023-03-14T11:23:13Z".toISODate()!.date
            ),
            ProductPrice(
                price: 8.9,
                supermarket: .aeon,
                date: "2023-03-13T11:23:13Z".toISODate()!.date
            ),
            ProductPrice(
                price: 28.9,
                supermarket: .aeon,
                date: "2023-03-12T11:23:13Z".toISODate()!.date
            ),
            ProductPrice(
                price: 8.9,
                supermarket: .aeon,
                date: "2023-03-20T11:23:13Z".toISODate()!.date
            ),
            ProductPrice(
                price: 25.0,
                supermarket: .dchfood,
                date: "2023-03-17T11:23:13Z".toISODate()!.date
            ),
            ProductPrice(
                price: 15.0,
                supermarket: .jasons,
                date: "2023-03-17T11:23:13Z".toISODate()!.date
            ),
            ProductPrice(
                price: 35.0,
                supermarket: .jasons,
                date: "2023-03-18T11:23:13Z".toISODate()!.date
            ),
            ProductPrice(
                price: 25.0,
                supermarket: .jasons,
                date: "2023-03-19T11:23:13Z".toISODate()!.date
            ),
            ProductPrice(
                price: 45.0,
                supermarket: .jasons,
                date: "2023-03-20T11:23:13Z".toISODate()!.date
            ),
            ProductPrice(
                price: 5.0,
                supermarket: .jasons,
                date: "2023-03-21T11:23:13Z".toISODate()!.date
            ),
            ProductPrice(
                price: 35.0,
                supermarket: .jasons,
                date: "2023-03-22T11:23:13Z".toISODate()!.date
            ),
            ProductPrice(
                price: 45.0,
                supermarket: .jasons,
                date: "2023-03-23T11:23:13Z".toISODate()!.date
            ),
            ProductPrice(
                price: 25.0,
                supermarket: .jasons,
                date: "2023-03-24T11:23:13Z".toISODate()!.date
            )],
        category1: "Drinks",
        category2: "Oriental drinks",
        category3: "Juices",
        photo: nil
    ),
    Product(
        name: "Coke - Can 330mL x 12",
        barcode: "4890008100385",
        nutrition: nil,
        manufacturer: nil,
        brand: "Coca Cola",
        prices: [
            ProductPrice(
                price: 44.0,
                supermarket: .aeon,
                date: "2023-03-16T11:23:13Z".toISODate()!.date
            ),
            ProductPrice(
                price: 49.9,
                supermarket: .aeon,
                date: "2023-03-19T11:23:13Z".toISODate()!.date
            ),
            ProductPrice(
                price: 35,
                supermarket: .dchfood,
                date: "2023-03-17T11:23:13Z".toISODate()!.date
            ),
            ProductPrice(
                price: 56,
                supermarket: .jasons,
                date: "2023-03-17T11:23:13Z".toISODate()!.date
            ),
            ProductPrice(
                price: 45,
                supermarket: .jasons,
                date: "2023-03-18T11:23:13Z".toISODate()!.date
            ),
            ProductPrice(
                price: 59,
                supermarket: .jasons,
                date: "2023-03-19T11:23:13Z".toISODate()!.date
            ),
            ProductPrice(
                price: 74.0,
                supermarket: .jasons,
                date: "2023-03-20T11:23:13Z".toISODate()!.date
            ),
            ProductPrice(
                price: 25.0,
                supermarket: .jasons,
                date: "2023-03-21T11:23:13Z".toISODate()!.date
            ),
            ProductPrice(
                price: 15.0,
                supermarket: .jasons,
                date: "2023-03-22T11:23:13Z".toISODate()!.date
            ),
            ProductPrice(
                price: 27.0,
                supermarket: .jasons,
                date: "2023-03-23T11:23:13Z".toISODate()!.date
            ),
            ProductPrice(
                price: 52.0,
                supermarket: .jasons,
                date: "2023-03-24T11:23:13Z".toISODate()!.date
            )],
        category1: "Drinks",
        category2: "Carbonated drinks",
        category3: "Canned",
        photo: nil
    ),
    Product(
        name: "Coke Zero - Bottle 500mL",
        barcode: "4890008109234",
        nutrition: nil,
        manufacturer: nil,
        brand: "Coca Cola",
        prices: [
            ProductPrice(
                price: 9.0,
                supermarket: .watsons,
                date: "2023-03-17T11:24:16Z".toISODate()!.date
            )
        ],
        category1: "Drinks",
        category2: "Carbonated drinks",
        category3: "Bottles",
        photo: nil
    ),
    Product(
        name: "Coke Plus (Zero Sugar) - Bottle 500mL",
        barcode: "4890008101238",
        nutrition: nil,
        manufacturer: nil,
        brand: "Coca Cola",
        prices: [
            ProductPrice(
                price: 29.0,
                supermarket: .watsons,
                date: "2023-03-17T11:24:16Z".toISODate()!.date
            )
        ],
        category1: "Drinks",
        category2: "Carbonated drinks",
        category3: "Bottles",
        photo: nil
    ),
    Product(
        name: "Coke Zero - Bottle 1.25L",
        barcode: "4890008109159",
        nutrition: nil,
        manufacturer: nil,
        brand: "Coca Cola",
        prices: [
            ProductPrice(
                price: 39.0,
                supermarket: .watsons,
                date: "2023-03-17T11:24:16Z".toISODate()!.date
            )
        ],
        category1: "Drinks",
        category2: "Carbonated drinks",
        category3: "Bottles",
        photo: nil
    ),
    Product(
        name: "Coke - Can 330mL x 8",
        barcode: "4890008100941",
        nutrition: .init(
            id: UUID().hashValue,
            energy: 20,
            protein: 200,
            total_fat: 20,
            saturated_fat: 200,
            trans_fat: 20,
            carbohydrates: 200,
            sugars: 20,
            sodium: 200,
            cholesterol: 20,
            vitaminB2: 200,
            vitaminB3: 20,
            vitaminB6: 200
        ),
        manufacturer: nil,
        brand: "Coca Cola",
        prices: [
            ProductPrice(
                price: 39.0,
                supermarket: .watsons,
                date: "2023-03-17T11:24:16Z".toISODate()!.date
            )
        ],
        category1: "Drinks",
        category2: "Carbonated drinks",
        category3: "Canned",
        photo: nil
    ),
    Product(
        name: "Coke Plus (Zero Sugar) - Bottle 500mL",
        barcode: "4890008101238",
        nutrition: .init(
            id: UUID().hashValue,
            energy: 10,
            protein: 100,
            total_fat: 10,
            saturated_fat: 100,
            trans_fat: 10,
            carbohydrates: 100,
            sugars: 10,
            sodium: 100,
            cholesterol: 10,
            vitaminB2: 100,
            vitaminB3: 10,
            vitaminB6: 100
        ),
        manufacturer: nil,
        brand: "Coca Cola",
        prices: [
            ProductPrice(
                price: 29.0,
                supermarket: .watsons,
                date: "2023-03-18T11:24:16Z".toISODate()!.date
            )
        ],
        category1: "Drinks",
        category2: "Carbonated drinks",
        category3: "Bottles",
        photo: nil
    )
]
let dummyMessages: [ChatBotMessage] = [
    ChatBotMessage(
        productEntity: "",
        clientInput: "",
        intentType: .productPrice,
        response: "I have found 1 product with the name lemon juice. You can have a look below." ,
        productsResponse: dummyProducts
    )
]

let dummyInventory: [Inventory] = [
    Inventory(
        name: "BBQ List",
        description: "My dummy list",
        products: dummyProducts
    )
]
