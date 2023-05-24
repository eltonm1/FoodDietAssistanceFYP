//
//  ChatBotPriceStatistics.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 18/3/2023.
//

import SwiftUI

struct ChatBotPriceStatistics: View {
    let low: Double
    let mean: Double
    let high: Double
    
    init(low: Double, mean: Double, high: Double) {
        self.low = low
        self.mean = mean
        self.high = high
    }
    
    init(prices: [ProductPrice]) {
        self.low = prices.min(by: { $0.price < $1.price })!.price
        self.mean = prices.reduce(0, { $0 + $1.price }) / Double(prices.count)
        self.high = prices.max(by: { $0.price < $1.price })!.price
    }
    
    var body: some View {
        HStack(spacing: 32) {
            VStack(alignment: .leading) {
                Text("Low")
                    .productFont(.regular, relativeTo: .body)
                    .foregroundColor(.secondary)
                Text("$\(low.formatted())")
                    .productFont(.bold, relativeTo: .title3)
                    .foregroundColor(.primary)
            }
            .background(alignment: .trailing) {
                Rectangle()
                    .foregroundColor(.secondary)
                    .offset(x: 16)
                    .frame(width: 2)
            }
            VStack(alignment: .leading) {
                Text("Mean")
                    .productFont(.regular, relativeTo: .body)
                    .foregroundColor(.secondary)
                Text("$\(mean.formatted())")
                    .productFont(.bold, relativeTo: .title3)
                    .foregroundColor(.primary)
            }
            .background(alignment: .trailing) {
                Rectangle()
                    .foregroundColor(.secondary)
                    .offset(x: 16)
                    .frame(width: 2)
            }
            VStack(alignment: .leading) {
                Text("High")
                    .productFont(.regular, relativeTo: .body)
                    .foregroundColor(.secondary)
                Text("$\(high.formatted())")
                    .productFont(.bold, relativeTo: .title3)
                    .foregroundColor(.primary)
            }
        }
    }
}

struct ChatBotPriceStatistics_Previews: PreviewProvider {
    static var previews: some View {
        ChatBotPriceStatistics(low: 10, mean: 20, high: 30)
    }
}
