//
//  CatagoryViewSearchSuggestion.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 20/3/2023.
//

import SwiftUI

extension CatagoryView {
    struct SearchSuggestion: View {
        @Namespace var ns
        @Binding var path: NavigationPath
        @Binding var search: String
        let products: [Product]
        let onClick: (Product) -> Void
        var onEnter: () -> Void
        var onLeave: () -> Void
        var body: some View {
            let search: String = search.lowercased()
            let results: [Product] = products
                .filter({
                    $0.name.lowercased().contains(search) ||
                    $0.barcode.contains(search)
                })
            ForEach(results) { (product: Product) in
                Button {
                    onClick(product)
                } label: {
                    ProductInformationRow(
                        ns: ns,
                        product: product,
                        color: .systemBlue,
                        onEnterInputView: onEnter,
                        onBackFromInputView: onLeave
                    )
                }
            }
            if !search.isEmpty && !results.isEmpty {
                Rectangle()
                    .fill(Color.clear)
                    .frame(height: 80)
            }
        }
    }
}
