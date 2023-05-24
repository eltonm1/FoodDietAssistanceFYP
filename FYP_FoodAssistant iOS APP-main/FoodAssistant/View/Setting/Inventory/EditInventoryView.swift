//
//  EditInventoryView.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 26/3/2023.
//

import SwiftUI

struct EditInventoryView: View {
    @ObservedObject var dataSource: FoodProductDataService
    @State var showSearchProductView: Bool = false
    @State var searchedProduct: Product?
    @State var similarProducts: [Product] = []
    @State fileprivate var sortBy: SortBy = .name
    @Binding var inventory: Inventory?
    @Binding var inventories: [Inventory]
    
    init(
        inventory: Binding<Inventory?>,
        inventories: Binding<[Inventory]>,
        dataSource: FoodProductDataService
    ) {
        self._inventory = inventory
        self._inventories = inventories
        self._dataSource = ObservedObject(wrappedValue: dataSource)
    }
    
    var body: some View {
        if let inventory {
            NavigationStack {
                List {
                    DetailSession(inventory: $inventory)
                    ProductsSession(
                        inventory: $inventory,
                        showSearchProductView: $showSearchProductView,
                        relocate: relocate,
                        sortBy: sortBy
                    )
                    SimilarProductsSession(
                        inventory: $inventory,
                        sortBy: $sortBy,
                        similarProducts: similarProducts
                    )
                }
                .navigationTitle(
                    inventory.name.isEmpty ? "Untitled Inventory" : inventory.name
                )
                .productLargeNavigationBar()
                .toolbar {
                    ToolBarMenu(
                        inventories: $inventories,
                        showSearchProductView: $showSearchProductView,
                        sortBy: $sortBy,
                        editingInventory: $inventory,
                        inventory: inventory
                    )
                }
                .sheet(isPresented: $showSearchProductView) {
                    SearchProductView(
                        selectedProduct: $searchedProduct,
                        dataSource: dataSource
                    )
                }
                .onChange(of: searchedProduct) { (newProduct: Product?) in
                    guard let newProduct: Product = newProduct else {
                        return
                    }
                    if inventory.products.contains(newProduct) {
                        return
                    }
                    self.inventory?.products.append(newProduct)
                    searchedProduct = nil
                }
                .task(id: inventory.products) {
                    guard !inventory.products.isEmpty else { return }
                    let productIDs: [String] = inventory.products.map(\.id.description.localizedLowercase)
                    guard let products: [Product] = try? await AppState
                        .shared
                        .dataService
                        .get(
                            type: [Product].self,
                            path: "/api/foodproducts/similarity/pkey/\(productIDs.joined(separator: "&"))/0"
                        ) else { return }
                    similarProducts = products
                }
            }
        }
    }
    
    func relocate(from source: IndexSet, to destination: Int) {
        inventory?.products.move(fromOffsets: source, toOffset: destination)
    }
}

fileprivate struct ToolBarMenu: View {
    @Environment(\.presentationMode) var dismiss
    @Binding var inventories: [Inventory]
    @Binding var showSearchProductView: Bool
    @Binding var sortBy: SortBy
    @Binding var editingInventory: Inventory?
    let inventory: Inventory
    var body: some View {
        if !inventory.products.isEmpty {
            EditButton()
                .productFont(.bold, relativeTo: .body)
        }
        Menu {
            Button(role: .destructive) {
                if let i: Int = inventories.firstIndex(of: inventory) {
                    inventories.remove(at: i)
                }
                editingInventory = nil
                dismiss.wrappedValue.dismiss()
            } label: {
                Label("Delete Inventory", systemImage: "trash")
            }
            Button {
                showSearchProductView.toggle()
            } label: {
                Label("Add Product", systemImage: "plus")
            }
            Menu {
                ForEach(SortBy.allCases, id: \.self) { (sb: SortBy) in
                    let isSelected: Bool = sortBy == sb
                    Button {
                        sortBy = sb
                        handleSortBy(sortBy: sb)
                    } label: {
                        Label("\(sb.rawValue)") {
                            if isSelected {
                                Image(systemName: "checkmark")
                            }
                        }
                    }
                }
            } label: {
                Label("Sort By", systemImage: "arrow.up.arrow.down")
            }
        } label: {
            Image(systemName: "ellipsis.circle")
        }
    }
    
    func handleSortBy(sortBy: SortBy) {
        switch sortBy {
        case .name:
            editingInventory?.products.sort { $0.name < $1.name }
        case .energy:
            editingInventory?.products.sort { ($0.nutrition?.energy ?? 0) < ($1.nutrition?.energy ?? 0) }
        case .sugar:
            editingInventory?.products.sort { ($0.nutrition?.sugars ?? 0) < ($1.nutrition?.sugars ?? 0) }
        case .carbohydrates:
            editingInventory?.products.sort { ($0.nutrition?.carbohydrates ?? 0) < ($1.nutrition?.carbohydrates ?? 0) }
        }
    }
}

fileprivate struct DetailSession: View {
    @Binding var inventory: Inventory?
    var body: some View {
        if let inventory {
            Section {
                ProductFontPlaceholderTextField(
                    text: Binding<String?>(
                        get: { inventory.name },
                        set: { self.inventory?.name = $0 ?? "Untitled Inventory" }
                    ),
                    placeholder: "My Product List"
                )
                ProductFontPlaceholderTextField(
                    text: Binding<String?>(
                        get: { inventory.description },
                        set: { self.inventory?.description = $0 }
                    ),
                    placeholder: "Description (Optional)"
                )
            } header: {
                Text("Inventory Detail")
                    .productFont(.regular, relativeTo: .footnote)
            }
        }
    }
}

fileprivate struct SimilarProductsSession: View {
    @Binding var inventory: Inventory?
    @Binding var sortBy: SortBy
    let similarProducts: [Product]
    func data(_ product: Product) -> Double {
        guard let n: NutritionInformation = product.nutrition else {
            return 0
        }
        return sortBy == .sugar ? n.sugars : sortBy == .energy ? Double(n.energy) : n.carbohydrates
    }
    var body: some View {
        if !similarProducts.isEmpty {
            Section {
                ForEach(
                    similarProducts
                        .sorted(by: { (a: Product, b: Product) in
                            switch sortBy {
                            case .name:
                                return a.name < b.name
                            case .energy:
                                return a.nutrition?.energy ?? 0 < b.nutrition?.energy ?? 0
                            case .sugar:
                                return a.nutrition?.sugars ?? 0 < b.nutrition?.sugars ?? 0
                            case .carbohydrates:
                                return a.nutrition?.carbohydrates ?? 0 < b.nutrition?.carbohydrates ?? 0
                            }
                        })
                        .sorted(by: { (a: Product, b: Product) in
                        isAdded(product: a) ? false : isAdded(product: b)
                    })
                ) { (product: Product) in
                    let added: Bool = isAdded(product: product)
                    HStack {
                        if !added {
                            Button {
                                inventory?.products.append(product)
                            } label: {
                                Image(systemName: "plus.circle.fill")
                                    .foregroundColor(.systemGreen)
                            }
                            .hoverEffect()
                        }
                        Text(product.name)
                            .productFont(.regular, relativeTo: .body)
                            .strikethrough(added, color: .systemRed)
                            .foregroundColor(.primary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        if sortBy != .name {
                            Text("\(data(product).formatted())g")
                                .productFont(.regular, relativeTo: .body)
                                .foregroundColor(.secondary)
                        }
                    }
                }
                .animation(.easeInOut, value: similarProducts)
            } header: {
                Text("Similar Products")
                    .productFont(.regular, relativeTo: .footnote)
            }
        }
    }
    
    func isAdded(product: Product) -> Bool {
        inventory?.products.contains(product) ?? false
    }
}

fileprivate struct ProductsSession: View {
    @Binding var inventory: Inventory?
    @Binding var showSearchProductView: Bool
    let relocate: (IndexSet, Int) -> Void
    let sortBy: SortBy
    var body: some View {
        if let inventory {
            Section {
                if inventory.products.isEmpty {
                    VStack {
                        Button {
                            showSearchProductView.toggle()
                        } label: {
                            Image(systemName: "plus.circle")
                                .resizable()
                                .scaledToFit()
                                .frame(width: 50)
                        }
                        Text("Add Products")
                            .productFont(.bold, relativeTo: .title3)
                            .foregroundColor(.primary)
                    }
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.vertical)
                } else {
                    ForEach(inventory.products) { (product: Product) in
                        HStack {
                            Text(product.name)
                                .productFont(.bold, relativeTo: .body)
                                .foregroundColor(.primary)
                            Spacer()
                            Group {
                                if sortBy == .sugar {
                                    Text("\((product.nutrition?.sugars ?? 0).formatted())g")
                                } else if sortBy == .energy {
                                    Text("\((product.nutrition?.energy ?? 0).formatted())kJ")
                                } else if sortBy == .carbohydrates {
                                    Text("\((product.nutrition?.carbohydrates ?? 0).formatted())g")
                                }
                            }
                            .productFont(.regular, relativeTo: .body)
                            .foregroundColor(.secondary)
                        }
                    }
                    .onMove(perform: relocate)
                    .onDelete { (iSet: IndexSet) in
                        self.inventory?.products.remove(atOffsets: iSet)
                    }
                    .animation(.easeInOut, value: inventory.products)
                }
            } header: {
                Text("Products")
                    .productFont(.regular, relativeTo: .footnote)
            }
        }
    }
}

fileprivate enum SortBy: String, CaseIterable {
    case name = "Name"
    case energy = "Energy"
    case sugar = "Sugar"
    case carbohydrates = "Carbohydrates"
}

struct EditInventoryView_Previews: PreviewProvider {
    @State static var inventory: Inventory? = Inventory()
    @State static var inventories: [Inventory] = []
    static var previews: some View {
        NavigationStack {
            EditInventoryView(
                inventory: $inventory,
                inventories: $inventories,
                dataSource: FoodProductDataService()
            )
        }
    }
}
