//
//  FoodProductService.swift
//  FoodAssistant
//
//  Created by Elton Miao on 28/12/2022.
//

import Foundation
import Combine
import SwiftUI

class FoodProductDataService: ObservableObject {
    
    @Published var clicked: Bool = true
    @Published var isLoading: Bool = true
    @Published var products: [Product] = []
    @Published var categories1: [String] = []
    @Published var categories2: [String] = []
    @Published var categories3: [String] = []
    @Published var errorMessage: String? = nil
    private var subscriptions = Set<AnyCancellable>()
    
    init() {
        $clicked
            .throttle(
                for: .seconds(1),
                scheduler: DispatchQueue.main,
                latest: true
            )
            .sink { [weak self] _ in
                self?.loadData()
            }
            .store(in: &subscriptions)
    }
    
    func loadData() {
//        isLoading = true
        AppState
            .shared
            .dataService
            .get(
                type: [Product].self,
                path: "/api/foodproducts/"
            )
            .receive(on: DispatchQueue.main)
            .sink { [weak self] (completion: Subscribers.Completion<Error>) in
                switch completion {
                case let .failure(error):
                    print("Couldn't load food products: \(error)")
                    withAnimation(.spring()) {
                        self?.errorMessage = error.localizedDescription
                    }
                case .finished:
                    self?.postProcessing()
                    self?.isLoading = false
                    break
                }
            } receiveValue: { [weak self] (products: [Product]) in
                self?.products = products.filter { (p: Product) in
                    !p.name.lowercased().contains("condom".lowercased())
                }
                print("Successfully loaded \(products.count) food products.")
            }
            .store(in: &subscriptions)
    }
    
    func addData(product: Product) {
        if AppState.shared.authService.user == nil { return }
        
        AppState
            .shared
            .dataService
            .post(
                object: product,
                type: Product.self,
                path: "/api/foodproducts/"
            )
            .receive(on: DispatchQueue.main)
            .sink { [weak self] (completion: Subscribers.Completion<Error>) in
                switch completion {
                case let .failure(error):
                    print("Couldn't add food product: \(error)")
                    self?.errorMessage = error.localizedDescription
                case .finished: break
                }
            } receiveValue: { [weak self] (product: Product) in
                self?.products.append(product)
            }
            .store(in: &subscriptions)
    }
    
    func putData(product: Product) {
        Task { [weak self] in
            guard let self = self else { return }
            let refinedProduct: Product = try await AppState
                .shared
                .dataService
                .put(
                    object: product,
                    type: Product.self, path: "/api/foodproducts/" + product.barcode)
            guard let id: Int = products.firstIndex(of: product) else {
                return
            }
            await MainActor.run { [weak self] in
                guard let self = self else { return }
                self.products[id] = refinedProduct
            }
        }
    }
    
    func postProcessing() {
        // Find unique value in products.category{1/2/3}
        self.categories1 = Array(Set(self.products.compactMap({ $0.category1 })))
        self.categories2 = Array(Set(self.products.compactMap({ $0.category2 })))
        self.categories3 = Array(Set(self.products.compactMap({ $0.category3 })))
    }
    
    // Return array of products whose categroy_{1,2,3} is categoryStr
    func productWhoweCategory(number: Int, is categoryStr: String) -> [Product] {
        if number == 1 {
            return products.filter {
                $0.category1 == categoryStr
            }
        } else if number == 2 {
            return products.filter {
                $0.category2 == categoryStr
            }
        } else {
            return products.filter {
                $0.category3 == categoryStr
            }
        }
    }

    func searchSimilarProducts(by barcode: String) -> [Product] {
        return products.filter {
            $0.barcode.contains(barcode)
        }
    }
    
    /// Update the given product with latest info
    ///
    /// - Parameters:
    ///     - oldProduct: The old product to be updated
    /// - Returns: The same product but with newest information
    func updateProduct(oldProduct: Product) -> Product? {
        guard let index: Int = products.firstIndex(of: oldProduct) else {
            return nil
        }
        return products[index]
    }
}
