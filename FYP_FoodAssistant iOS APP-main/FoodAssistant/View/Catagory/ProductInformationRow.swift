//
//  ProductInformationRow.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 14/3/2023.
//

import SwiftUI

struct ProductInformationRow: View {
    @Environment(\.colorScheme) var scheme
    @EnvironmentObject var mvm: MainViewModel
    let ns: Namespace.ID
    let product: Product
    let color: Color
    let onEnterInputView: () -> Void
    let onBackFromInputView: () -> Void
    
    var detail: InputProductDetailView.Detail {
        InputProductDetailView.Detail(
            product: product,
            onAppear: onEnterInputView,
            onDisappear: onBackFromInputView,
            onUpload: mvm.foodDataService.putData
        )
    }
    
    var body: some View {
        HStack(spacing: 16) {
            Image(systemName: "fork.knife.circle")
                .foregroundColor(.white)
                .padding(6)
                .background(color)
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
        .matchedGeometryEffect(id: "\(product.barcode)-\(product.name)", in: ns)
        .previewContextMenu(
            destination: InputProductDetailView(detail: detail),
            preview: InputProductDetailView(
                detail: InputProductDetailView.Detail(product: product)
            ),
            navigationValue: CatagoryViewModel
                .NavigationRoute
                .inputProductDetailView(detail),
            presentAsSheet: false
        )
        .padding(.horizontal)
    }
}
