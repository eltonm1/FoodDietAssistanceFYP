//
//  SearchProductView.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 26/3/2023.
//

import SwiftUI

struct SearchProductView: View {
    @Binding var selectedProduct: Product?
    @ObservedObject var dataSource: FoodProductDataService
    @Environment(\.colorScheme) var scheme
    @Environment(\.presentationMode) var dismiss
    @State var searchingText: String = ""
    var filteredProducts: [Product] {
        searchingText.isEmpty ? dataSource.products : dataSource
            .products
            .filter {
                $0.barcode.contains(
                    searchingText.lowercased()
                ) ||
                $0.name.lowercased().contains(
                    searchingText.lowercased()
                )
            }
    }
    var body: some View {
        NavigationStack {
            List(filteredProducts) { (product: Product) in
                Button {
                    selectedProduct = product
                    dismiss.wrappedValue.dismiss()
                } label: {
                    HStack(spacing: 16) {
                        Image(systemName: "fork.knife.circle")
                            .foregroundColor(.white)
                            .padding(6)
                            .background(.random)
                            .clipShape(Circle())
                        VStack(alignment: .leading) {
                            Text(product.name)
                                .multilineTextAlignment(.leading)
                                .foregroundColor(.primary)
                                .productFont(.bold, relativeTo: .title3)
                            HStack(alignment: .top) {
                                Image(systemName: "barcode")
                                    .foregroundColor(.secondary)
                                Text("Barcode: \(product.barcode)")
                                    .foregroundColor(.secondary)
                                    .productFont(.regular, relativeTo: .body)
                                    .lineLimit(2)
                                    .multilineTextAlignment(.leading)
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        Text(
                            product.prices.first?.price == nil ?
                            "NA" : "$\(product.prices.first!.price.formatted())"
                        )
                        .foregroundColor(.primary)
                        .productFont(.bold, relativeTo: .body)
                        .padding(8)
                        .background(.secondary.opacity(scheme == .dark ? 0.4 : 0.2))
                        .clipShape(Capsule())
                    }
                }
            }
            .searchable(text: $searchingText, prompt: "e.g. Coke - Bottle 1.25L")
        }
    }
}

struct SearchProductView_Previews: PreviewProvider {
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
