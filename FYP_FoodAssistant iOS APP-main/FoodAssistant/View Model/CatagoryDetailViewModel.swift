//
//  CatagoryDetailViewModel.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 3/1/2023.
//

import Foundation
import SwiftUI

class CatagoryDetailViewModel: ObservableObject {
    
    // MARK: Character Model For Holding Data about Each Alphabet
    class ProductCharacter: IdentifyEquateCodeHashable {
        var id = UUID()
        var value: String
        var index: Int = 0
        var minY: CGFloat = .zero
        var pusOffset: CGFloat = 0
        var isCurrent: Bool = false
        var products: [Product] = []

        init(value: String) {
            self.value = value
        }
    }
    
    enum SortBy: CaseIterable {
        case price, name
        
        var labelName: String {
            switch self {
            case .price: return "Price"
            case .name: return "Name"
            }
        }
        
        var systemImage: String {
            switch self {
            case .price: return "dollarsign"
            case .name: return "abc"
            }
        }
    }
    
    enum OrderBy: CaseIterable {
        case ascending, descending
        
        var labelName: String {
            switch self {
            case .ascending: return "Ascending"
            case .descending: return "Descending"
            }
        }
        
        var systemImage: String {
            switch self {
            case .ascending: return "arrow.up"
            case .descending: return "arrow.down"
            }
        }

    }
    
    @Published var searchedProduct = ""
    @Published var characters: [ProductCharacter]
    @Published var scrollerHeight: CGFloat = 0
    @Published var startOffset: CGFloat = 0
    @Published var hideIndicatorLabel: Bool = false
    @Published var currentCharacter: ProductCharacter? = nil
    @Published var indicatorOffset: CGFloat = 0
    var scrollerTimeOut: CGFloat = 0.3
    @Published var sortBy: SortBy = .price
    @Published var orderBy: OrderBy = .ascending
    @Published var expandedCharacters: [Int] = []
    let products: [Product]
    
    init(products: [Product]) {
        self.products = products
        
        let alphabet: [String] = Array(Set(products.map { (p: Product) in
            String(p.name.prefix(1))
        })).sorted()
        self.characters = alphabet.map { (s: String) in
            ProductCharacter(value: s)
        }
        for index in 0..<self.characters.count {
            self.characters[index].index = index
            self.characters[index].products = products.filter { (p: Product) in
                p.name.lowercased().starts(with: self.characters[index].value.lowercased())
            }
        }
        expandedCharacters = Array(0..<characters.count)
    }
    
    var scrollerTitle: String {
        currentCharacter?.value ?? ""
    }
    
    var scrollSubtitle: String {
        "\(currentCharacter?.products.count ?? 0) items"
    }
    
    var scrollerShouldDisappear: Bool {
        hideIndicatorLabel || currentCharacter == nil
    }

    var toolBarItem: MenuItem {
        MenuItem(systemName: "ellipsis.circle") {
            MenuItem(label: "Sort By", systemName: "arrow.left.arrow.right") {
                SortBy
                    .allCases
                    .map { (sorting: SortBy) in
                        ButtonItem(label: sorting.labelName, systemName: sorting.systemImage) {
                            withAnimation(.spring()) {
                                self.sortList(by: sorting)
                            }
                        }
                    }
            }
            MenuItem(label: "Order By", systemName: "arrow.up.arrow.down") {
                OrderBy
                    .allCases
                    .map { (ordering: OrderBy) in
                        ButtonItem(label: ordering.labelName, systemName: ordering.systemImage) {
                            withAnimation(.spring()) {
                                self.orderList(by: ordering)
                            }
                        }
                    }
            }
            ButtonItem(label: "Expand All", systemName: "rectangle.expand.vertical") {
                self.expandAll()
            }
            ButtonItem(label: "Compress All", systemName: "rectangle.compress.vertical") {
                self.compressAll()
            }
        }
    }
    
    var expandedDict: [ProductCharacter : Bool] {
        Dictionary(uniqueKeysWithValues: self.characters.map { (pc: ProductCharacter) in
            (pc, self.isExpanded(pc: pc))
        })
    }

    var filteredProductsDict: [ProductCharacter : [Product]] {
        Dictionary(uniqueKeysWithValues: self.characters.map { (pc: ProductCharacter) in
            (pc, self.filteredProducts(in: pc))
        })
    }
    
    func filteredProducts(in pc: ProductCharacter) -> [Product] {
        if self.searchedProduct.isEmpty {
            return pc.products
        } else {
            return pc.products.filter { (p: Product) in
                p.name.lowercased().contains(self.searchedProduct.lowercased())
            }
        }
    }
    
    func sortList(by sorting: SortBy) {
        switch sorting {
        case .name:
            self.characters = self.characters.map { (pc: ProductCharacter) in
                pc.products.sort { (p1: Product, p2: Product) in
                    switch self.orderBy {
                    case .ascending:
                        return p1.name < p2.name
                    case .descending:
                        return p1.name > p2.name
                    }
                }
                return pc
            }
        case .price:
            self.characters = self.characters.map { (pc: ProductCharacter) in
                pc.products.sort { (p1: Product, p2: Product) in
                    switch self.orderBy {
                    case .ascending:
                        return (p1.prices[safe: 0]?.price ?? 0) < (p2.prices[safe: 0]?.price ?? 0)
                    case .descending:
                        return (p1.prices[safe: 0]?.price ?? 0) > (p2.prices[safe: 0]?.price ?? 0)
                    }
                }
                return pc
            }
        }
    }
    
    func orderList(by ordering: OrderBy) {
        switch ordering {
        case .ascending:
            self.characters = self.characters.map { (pc: ProductCharacter) in
                pc.products.sort { (p1: Product, p2: Product) in
                    switch self.sortBy {
                    case .name:
                        return p1.name < p2.name
                    case .price:
                        return (p1.prices[safe: 0]?.price ?? 0) < (p2.prices[safe: 0]?.price ?? 0)
                    }
                }
                return pc
            }
        case .descending:
            self.characters = self.characters.map { (pc: ProductCharacter) in
                pc.products.sort { (p1: Product, p2: Product) in
                    switch self.sortBy {
                    case .name:
                        return p1.name > p2.name
                    case .price:
                        return (p1.prices[safe: 0]?.price ?? 0) > (p2.prices[safe: 0]?.price ?? 0)
                    }
                }
                return pc
            }
        }
    }
    
    func toggleExpand(of pc: ProductCharacter) {
        withAnimation(.spring()) {
            if isExpanded(pc: pc) {
                expandedCharacters.removeAll { (i: Int) in
                    i == pc.index
                }
            } else {
                expandedCharacters.append(pc.index)
            }
        }
    }
    
    func isExpanded(pc: ProductCharacter) -> Bool {
        expandedCharacters.contains(pc.index)
    }
    
    func expandAll() {
        expandedCharacters = Array(0..<characters.count)
    }
    
    func compressAll() {
        withAnimation(.spring()) {
            expandedCharacters = []
        }
    }
    
    func onNavigateToInputView(mvm: MainViewModel, isEntering: Bool) {
        withAnimation(.spring()) {
            mvm.bottomBarVM.setSrollable(to: !isEntering)
            mvm.bottomBarVM.showBar = !isEntering
        }
    }
    
    func onTimerUpdate() {
        if scrollerTimeOut < 2 {
            scrollerTimeOut += 0.01
        } else {
            // MARK: Scrolling is Finished
            // It Will Fire Many Times So Use Some Conditions Here
            if !hideIndicatorLabel {
                // Scrolling is Finished
                hideIndicatorLabel = true
            }
        }
    }
}
