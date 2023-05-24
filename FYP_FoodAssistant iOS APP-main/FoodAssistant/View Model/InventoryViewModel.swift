//
//  InventoryViewModel.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 25/3/2023.
//

import Foundation
import Combine

class InventoryViewModel: ObservableObject {
    @Published var foodDataService: FoodProductDataService
    @Published var inventories: [Inventory]
    @Published var searchingText: String
    @Published var editingInventory: Inventory?
    @Published var summaryType: SummaryCategory? = nil
    
    var anyCancellables = Set<AnyCancellable>()
    init(dataSource: FoodProductDataService) {
        searchingText = ""
        /// Note: Check if there is any inventory in the user defaults
        /// If there is, load it, otherwise, create a new one
        do {
            if let data: Data = UserDefaults.standard.object(forKey: "inventories") as? Data {
               let inventories: [Inventory] = try JSONDecoder().decode([Inventory].self, from: data)
                self.inventories = inventories
            } else {
                self.inventories = []
            }
        } catch {
            print(error)
            self.inventories = []
        }
        
        self._foodDataService = Published(wrappedValue: dataSource)
        foodDataService.objectWillChange.sink { [weak self] in
            guard let self = self else { return }
            self.updateInventories()
            self.objectWillChange.send()
        }
        .store(in: &anyCancellables)
    }
    
    func updateInventories() {
        if foodDataService.products.isEmpty {
            return
        }
        /// - Note: Since Inventories are stored in user default, which maybe outdated
        /// in term of product information, so everytime user launch the page,
        /// we need to refetch the products to update them with latest information
        for (index, inv) in self.inventories.enumerated() {
            /// - Note: using compactMap because if the locally stored product is not present in database
            /// that means the product is possibly deleted recently, so we need to sync this deletion here
            self.inventories[index].products = inv.products.compactMap(self.foodDataService.updateProduct)
        }
        if let encoded: Data = try? JSONEncoder().encode(inventories) {
            UserDefaults.standard.set(encoded, forKey: "inventories")
        }
    }
    
    deinit {
        updateInventories()
    }
}

extension InventoryViewModel {
    enum SummaryCategory: String, CaseIterable {
        case energy = "Energy"
        case sugar = "Sugar"
        case carbohydrates = "Carbohydrates"
    }
}
