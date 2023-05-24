//
//  InputProductDetailViewModel.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 17/1/2023.
//

import Combine
import SwiftUI

class InputProductDetailViewModel: ObservableObject {
    @EnvironmentObject var mvm: MainViewModel
    let product: Product
    let boundingBox: BoundingBox?
    let nutritionTablePhoto: Photo?
    @Published var barcode: String?
    @Published var name: String?
    @Published var price: Double?
    @Published var brand: String?
    @Published var energy: String?
    @Published var protein: String?
    @Published var totalFat: String?
    @Published var saturatedFat: String?
    @Published var transFat: String?
    @Published var carbohydrates: String?
    @Published var sugars: String?
    @Published var sodium: String?
    @Published var vitaminB2: String?
    @Published var vitaminB3: String?
    @Published var vitaminB6: String?
    @Published var showDismissAlert: Bool = false
    @Published var hasEditedAnything: Bool = false
    
    let editable: Bool
    let onUpload: ((Product) -> Void)?
    var cancellables = Set<AnyCancellable>()
    
    init(
        product: Product,
        boundingBox: BoundingBox?,
        nutritionTablePhoto: Photo?,
        editable: Bool,
        onUpload: ((Product) -> Void)?
    ) {
        self.product = product
        self.boundingBox = boundingBox
        self.nutritionTablePhoto = nutritionTablePhoto
        self.editable = editable
        self.onUpload = onUpload
        
        barcode = product.barcode
        name = product.name
        price = product.prices.first?.price
        brand = product.brand
        energy = product.nutrition?.energy == nil ? nil : String(product.nutrition!.energy)
        protein = product.nutrition?.protein == nil ? nil : String(product.nutrition!.protein)
        totalFat = product.nutrition?.total_fat == nil ? nil : String(product.nutrition!.total_fat)
        saturatedFat = product.nutrition?.saturated_fat == nil ? nil : String(product.nutrition!.saturated_fat)
        transFat = product.nutrition?.trans_fat == nil ? nil : String(product.nutrition!.trans_fat)
        carbohydrates = product.nutrition?.carbohydrates == nil ? nil : String(product.nutrition!.carbohydrates)
        sugars = product.nutrition?.sugars == nil ? nil : String(product.nutrition!.sugars)
        sodium = product.nutrition?.sodium == nil ? nil : String(product.nutrition!.sodium)
        vitaminB2 = product.nutrition?.vitaminB2 == nil ? nil : String(product.nutrition!.vitaminB2!)
        vitaminB3 = product.nutrition?.vitaminB3 == nil ? nil : String(product.nutrition!.vitaminB3!)
        vitaminB6 = product.nutrition?.vitaminB6 == nil ? nil : String(product.nutrition!.vitaminB6!)
    }
    
    func onAnyTextFieldChanged() {
        self.hasEditedAnything = true
    }
    
    func onScanNutTableButTap() {
        
    }
    
    func uploadProductInformationToServer() {
//        guard editable else { return }
        let information: NutritionInformation = .init(
            id: product.nutrition?.id ?? 0,
            energy: Int(energy ?? "") ?? 0,
            protein: Double(protein ?? "") ?? 0,
            total_fat: Double(totalFat ?? "") ?? 0,
            saturated_fat: Double(saturatedFat ?? "") ?? 0,
            trans_fat: Double(transFat ?? "") ?? 0,
            carbohydrates: Double(carbohydrates ?? "") ?? 0,
            sugars: Double(sugars ?? "") ?? 0,
            sodium: Double(sodium ?? "") ?? 0,
            cholesterol: product.nutrition?.cholesterol ?? 0,
            vitaminB2: Double(vitaminB2 ?? ""),
            vitaminB3: Double(vitaminB3 ?? ""),
            vitaminB6: Double(vitaminB6 ?? "")
        )
        let product: Product = .init(
            id: product.id,
            name: name ?? "",
            barcode: barcode ?? "",
            nutrition: information,
            manufacturer: nil,
            brand: product.brand,
            prices: [],
            category1: product.category1,
            category2: product.category2,
            category3: product.category3,
            photo: product.photo
        )
        onUpload?(product)
    }
}
