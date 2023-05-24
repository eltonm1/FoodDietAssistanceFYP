//
//  ChatBotPriceMultipleProductsCell.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 18/3/2023.
//

import SwiftUI
import Charts
import SwiftDate

extension ChatBotSpecialMessageView.PriceCell {
    struct MultipleProductsCell: View {
        @Environment(\.colorScheme) var scheme
        @Binding var path: NavigationPath
        let supermarkets: [Supermarket]
        let products: [Product]
        let viewWidth: CGFloat
        
        init(products: [Product], viewWidth: CGFloat, path: Binding<NavigationPath>) {
            self.products = products
            self.supermarkets = Array(Set(products.flatMap(\.prices).map(\.supermarket))).sorted(by: \.rawValue)
            self.viewWidth = viewWidth
            self._path = path
        }
        
        var body: some View {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 16) {
                    ForEach(supermarkets, id: \.self) { (sm: Supermarket) in
                        VStack(alignment: .leading) {
                            Text("\(sm.rawValue) Store")
                                .productFont(.bold, relativeTo: .body)
                                .foregroundColor(.secondary)
                            let productsThatHaveOnlyOneDayPrice: [Product] = products.filter {
                                $0.prices.filter( { $0.supermarket == sm } ).count == 1
                            }
                            let productsThatHaveMoreThanOneDayPrices: [Product] = products.filter {
                                $0.prices.filter( { $0.supermarket == sm } ).count > 1
                            }
                            if !productsThatHaveOnlyOneDayPrice.isEmpty {
                                SingleDayCard(
                                    path: $path,
                                    products: productsThatHaveOnlyOneDayPrice,
                                    sm: sm
                                )
                                .equatable(by: productsThatHaveOnlyOneDayPrice)
                            } else if !productsThatHaveMoreThanOneDayPrices.isEmpty {
                                MultiDaysCard(
                                    path: $path,
                                    products: productsThatHaveMoreThanOneDayPrices,
                                    sm: sm
                                )
                                .equatable(by: productsThatHaveMoreThanOneDayPrices)
                            }
                        }
                        .padding(32)
                        .frame(width: max(0, viewWidth - 48), height: 400)
                        .background(Color.black.brightness(scheme == .light ? 0.95 : 0.2))
                        .cornerRadius(20, style: .continuous)
                    }
                }
                .padding(.horizontal)
            }
        }
    }
}

struct ChatBotPriceMultipleProductsCell_Previews: PreviewProvider {
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
