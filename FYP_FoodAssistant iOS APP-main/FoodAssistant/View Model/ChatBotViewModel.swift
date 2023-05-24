//
//  ChatBotViewModel.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 14/3/2023.
//

import Combine
import SwiftUI

class ChatBotViewModel: ObservableObject {
    @Published var dataSource: FoodProductDataService?
    @Published var message: String?
    @Published var cachedProductEntity: String = ""
    @Published var messages: [ChatBotMessage] = []
    @Published var fullViewRectExcludingNavArea: CGRect = .zero
    @Published var navigationPath = NavigationPath()
    
    var anyCancellables = Set<AnyCancellable>()
    
    var startOffset: CGFloat {
        fullViewRectExcludingNavArea.minY
    }
    
    init(dataSource: FoodProductDataService? = nil) {
        if let dataSource {
            self._dataSource = Published(wrappedValue: dataSource)
            
            dataSource.objectWillChange.sink { [weak self] (_) in
                self?.objectWillChange.send()
            }
            .store(in: &anyCancellables)
        }
    }
    
    func sendMessage() {
        guard let message: String = message, !message.isEmpty else { return }
        self.message = nil
        messages.append(
            ChatBotMessage(
                productEntity: "",
                clientInput: message
            )
        )
        Task {
            do {
                if(message ==  ""){
                    return
                }
                let responseMessage: ChatBotMessage = try await AppState
                    .shared
                    .dataService
                    .post(object:
                            ChatBotMessage(
                                productEntity: cachedProductEntity,
                                clientInput: message
                            ),
                          type: ChatBotMessage.self,
                          host: "20.187.76.166",
                          port: 9999,
                          path: "/chatbot"
                    )
                await MainActor.run {
                    self.message = nil
                    self.cachedProductEntity = responseMessage.productEntity
                    self.messages.append(responseMessage)
                }
            } catch {
                print(error)
            }
        }
    }
    
    func onRootViewRectChange(rect: CGRect) {
//        if startOffset == 0 {
//            print("start offset: \(rect.minY), rect: \(rect)")
            fullViewRectExcludingNavArea = rect
//        }
    }
}
