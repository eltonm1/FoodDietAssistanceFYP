//
//  ChatBotSingleProductSingleDayCard.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 20/3/2023.
//

import SwiftUI
import Charts

extension ChatBotSpecialMessageView.PriceCell.SingleProductCell {
    struct SingleDayCard: View {
        let price: ProductPrice
        let name: String
        var body: some View {
            VStack {
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
                    .frame(maxHeight: .infinity, alignment: .center)
                Text("\(name)")
                    .productFont(.bold, relativeTo: .title2)
                    .multilineTextAlignment(.center)
                    .foregroundColor(.secondary)
            }
            .frame(maxWidth: .infinity)
        }
    }
}

struct ChatBotSingleProductCellSingleDayCard_Previews: PreviewProvider {
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
