//
//  SimilarProductsView.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 6/3/2023.
//

import SwiftUI

struct SimilarProductView: View {
    let products: [Product]
    
    init(
        products: [Product] = [
            .init(name: "hi", barcode: "456", nutrition: nil, manufacturer: "yes", brand: "Yes", prices: [], category1: "1", category2: "2", category3: "3")
        ]
    ) {
        self.products = products
    }
    
    var body: some View {
        if products.isEmpty {
            GeometryReader { (proxy: GeometryProxy) in
                let size: CGSize = proxy.size
                VStack {
                    Image(systemName: "rectangle.and.text.magnifyingglass")
                        .resizable()
                        .scaledToFit()
                        .frame(width: size.width/3)
                        .foregroundColor(.primary)
                    Text("Oops")
                        .productFont(.bold, relativeTo: .largeTitle)
                        .foregroundColor(.primary)
                    Text("Seems like there isn't any match for this barcode in our server : (")
                        .multilineTextAlignment(.center)
                        .productFont(.regular, relativeTo: .title3)
                        .foregroundColor(.secondary)
                }
                .frame(width: size.width, height: size.height)
            }
            .padding(32)
        } else {
            List(products) { (product: Product) in
                ProductRow(product: product)
            }
        }
    }
}

fileprivate struct ProductRow: View {
    let product: Product
    var body: some View {
        Text(product.name)
    }
}

struct SimilarProductView_Preview: PreviewProvider {
    static var previews: some View {
        SimilarProductView()
    }
}
