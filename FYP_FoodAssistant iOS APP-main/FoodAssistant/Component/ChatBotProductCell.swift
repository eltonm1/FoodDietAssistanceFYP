//
//  ChatBotProductCell.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 20/3/2023.
//

import SwiftUI

struct ChatBotProductCell: View {
    @Environment(\.colorScheme) var scheme
    let product: Product
    let color: Color
    let width: CGFloat
    let height: CGFloat
    
    var detail: InputProductDetailView.Detail {
        InputProductDetailView
            .Detail(
                product: product,
                editable: false
            )
    }
    
    var body: some View {
        NavigationLink(value: detail) {
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
                            .lineLimit(1)
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
            .padding(.horizontal)
            .frame(width: max(0, width), height: height)
            .background(Color.black.brightness(scheme == .light ? 0.95 : 0.2))
            .cornerRadius(20, style: .continuous)
        }
        .previewContextMenu(
            destination: InputProductDetailView(detail: detail),
            preview: InputProductDetailView(detail: detail),
            navigationValue: detail,
            presentAsSheet: false
        )
    }
}

struct ChatBotProductCell_Previews: PreviewProvider {
    static var previews: some View {
        GeometryReader {
            ChatBotProductCell(
                product: dummyProducts[0],
                color: .random,
                width: max(0, $0.size.width - 48),
                height: 100
            )
            .frame(width: $0.size.width, alignment: .center)
        }
    }
}
