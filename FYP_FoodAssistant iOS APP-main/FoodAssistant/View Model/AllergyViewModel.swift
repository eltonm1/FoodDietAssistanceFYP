//
//  AllergyViewModel.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 6/1/2023.
//

import Foundation
import SwiftUI

class AllergyViewModel: ObservableObject {
    @Published var selectedAllergies: [Ingredient] = []
    @Published var remainingAllergies: [Ingredient] = Ingredient.allCases.sorted(by: <)
    
    func addAllergy(ingredient: Ingredient) {
        withAnimation {
            if !selectedAllergies.contains(ingredient) {
                selectedAllergies.append(ingredient)
                selectedAllergies = selectedAllergies.sorted(by: <)
            }
            remainingAllergies.removeAll { (i: Ingredient) in
                i == ingredient
            }
        }
    }
    
    func removeAllergy(ingredient: Ingredient) {
        withAnimation {
            selectedAllergies.removeAll { (i: Ingredient) in
                i == ingredient
            }
            if !remainingAllergies.contains(ingredient) {
                remainingAllergies.append(ingredient)
                remainingAllergies = remainingAllergies.sorted(by: <)
            }
        }
    }
}
