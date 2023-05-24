//
//  ChatBotPriceMultiProductsSingleSameDaySingleProductCard.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 20/3/2023.
//

import SwiftUI
import Charts

extension ChatBotSpecialMessageView.PriceCell.MultipleProductsCell.SingleDayCard {
    struct SingleProductCard: View {
        let product: Product
        let detail: InputProductDetailView.Detail
        init(product: Product) {
            self.product = product
            self.detail = .init(product: product, editable: false)
        }
        var body: some View {
            VStack {
                let price: ProductPrice = product.prices.first!
                Text("On \(price.date.formatted(.dateTime.day().month().year()))")
                    .productFont(.bold, relativeTo: .title2)
                    .foregroundColor(.primary)
                Text("$\(price.price.formatted())")
                    .productFont(.bold, relativeTo: .largeTitle)
                    .foregroundColor(.primary)
                    .padding()
                    .padding(.horizontal)
                    .background(.systemBlue)
                    .clipShape(Capsule())
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
                Text("\(product.name)")
                    .productFont(.bold, relativeTo: .title2)
                    .multilineTextAlignment(.center)
                    .foregroundColor(.primary)
            }
            .previewContextMenu(
                destination: InputProductDetailView(detail: detail),
                preview: InputProductDetailView(detail: detail),
                navigationValue: detail,
                presentAsSheet: false
            )
        }
    }
}

struct ChatBotPriceMultiProductsSingleSameDaySingleProductCard_Previews: PreviewProvider {
    @State static var path = NavigationPath()
    static var previews: some View {
        GeometryReader { (proxy: GeometryProxy) in
            let width: CGFloat = proxy.size.width
            let height: CGFloat = proxy.size.height
            ChatBotSpecialMessageView
                .PriceCell(message: dummyMessages[0], viewWidth: width, path: $path)
                .frame(height: height, alignment: .center)
        }
        .frame(maxHeight: .infinity, alignment: .center)
        .preferredColorScheme(.dark)
    }
}
