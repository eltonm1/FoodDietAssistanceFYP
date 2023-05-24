//
//  ProductViewModel.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 18/11/2022.
//

import Foundation
import SwiftUI
import Combine

class CatagoryViewModel: ObservableObject {
    
    enum NavigationRoute: Hashable {
        case categoryDetailView(CatagoryDetailView.CategoryDetail)
        case inputProductDetailView(InputProductDetailView.Detail)
    }
    
    enum ViewType: CaseIterable {
        case gallery, list
        
        var label: String {
            switch self {
            case .gallery: return "List"
            case .list: return "Gallery"
            }
        }
        
        var systemName: String {
            switch self {
            case .gallery: return "list.bullet"
            case .list: return "square.grid.2x2"
            }
        }
    }
    
    @Published private var _searchedCatagory = ""
    @Published var foodsService: FoodProductDataService
    @Published var viewType: ViewType = .list
    @Published var colors: [String : Color] = [:]
    @Published var navigationPath = NavigationPath()
    
    var searchedCatagory: Binding<String> {
        Binding<String>(get: {
            self._searchedCatagory
        }, set: { (s: String) in
            withAnimation(.spring()) {
                self._searchedCatagory = s
            }
        })
    }
    
    var anyCancellables = Set<AnyCancellable>()
    
    var filteredCats: [String] {
        if _searchedCatagory.isEmpty {
            return foodsService.categories1
        } else {
            return foodsService.categories1.filter { (cat: String) in
                cat.lowercased().contains(_searchedCatagory.lowercased())
            }
        }
    }
    
    var catProductsDict: [String : [Product]] {
        Dictionary(uniqueKeysWithValues: self.filteredCats.map {
            (cat: String) -> (String, [Product]) in
            (cat, self.foodsService.productWhoweCategory(number: 1, is: cat))
        })
    }
    
    var toolBarItem: MenuItem {
        MenuItem(systemName: "ellipsis.circle") {
            ButtonItem(label: "View as \(viewType.label)", systemName: viewType.systemName) {
                withAnimation(.spring()) {
                    self.toggleViewType()
                }
            }
            ButtonItem(label: "Add Product", systemName: "plus") {
                /// TODO: Navigate user to `CameraView` and prompt a message asking user
                /// to scan the barcode/product in that view
            }
        }
    }
    
    init(foodService: FoodProductDataService) {
        self._foodsService = Published(wrappedValue: foodService)
        foodsService.objectWillChange.sink { [weak self] _ in
            self?.objectWillChange.send()
        }
        .store(in: &anyCancellables)
        
        foodsService.$categories1.sink { [weak self] (cats: [String]) in
            self?.colors = Dictionary(uniqueKeysWithValues: cats.map { (cat: String) in
                (cat, Color.random)
            })
        }
        .store(in: &anyCancellables)
    }
    
    func toggleViewType() {
        if viewType == .list {
            viewType = .gallery
        } else {
            viewType = .list
        }
    }
    
    func onSearchSuggestionClicked(product: Product) {
        let detail = InputProductDetailView.Detail(
            product: product,
            editable: false
        )
        navigationPath.append(
            CatagoryViewModel
                .NavigationRoute
                .inputProductDetailView(detail)
        )
        
    }
    
    func onNavigateToInputView(mvm: MainViewModel, isEntering: Bool) {
        withAnimation(.spring()) {
            mvm.bottomBarVM.setSrollable(to: !isEntering)
            mvm.bottomBarVM.showBar = !isEntering
        }
    }
    
    func onRefresh() {
        foodsService.loadData()
    }
    
}
