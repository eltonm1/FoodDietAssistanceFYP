//
//  ChatBotView.swift
//  FoodAssistant
//
//  Created by Elton Miao on 9/3/2023.
//

import SwiftUI
import SwiftUIX

struct ChatBotView: View {
    @Environment(\.safeAreaInsets) var inset
    @Namespace var ns
    @StateObject var vm: ChatBotViewModel
    @StateObject var keyboard = Keyboard()
    @FocusState var messageFieldIsFocused: Bool
    
    init(dataSource: FoodProductDataService? = nil) {
        self._vm = StateObject(
            wrappedValue: ChatBotViewModel(
                dataSource: dataSource
            )
        )
    }
    
    @Environment(\.dismiss) var dismiss
    var body: some View {
        NavigationStack(path: $vm.navigationPath) {
            ScrollViewReader { (svProxy: ScrollViewProxy) in
                ScrollView {
                    VStack {
                        ForEach(vm.messages) { (message: ChatBotMessage) in
//                        ForEach(dummyMessages) { (message: ChatBotMessage) in
                            ChatBotMessageView(
                                path: $vm.navigationPath,
                                ns: ns,
                                viewWidth: max(0, vm.fullViewRectExcludingNavArea.width),
                                message: message
                            )
                            .tag(message)
                            .onTapGesture {
                                withAnimation(.easeOut(duration: 0.5)) {
                                    messageFieldIsFocused = false
                                    svProxy.scrollTo("empty", anchor: .bottom)
                                }
                            }
                        }
                        Rectangle()
                            .fill(.clear)
                            .frame(height: inset.bottom + 50)
                            .id("empty")
                    }
                    .onChange(of: vm.messages.count) { _ in
                        withAnimation(.easeOut(duration: 0.5)) {
                            svProxy.scrollTo("empty")
                        }
                    }
                }
                .onTapGesture {
                    messageFieldIsFocused = false
                }
            }
            .navigationTitle("Chat Bot")
            .productLargeNavigationBar()
            .coordinateSpace(name: "rootView")
            .overlay(alignment: .bottom) {
                ChatBotInputField(
                    message: $vm.message,
                    isFocused: _messageFieldIsFocused,
                    onSendButtonTapped: vm.sendMessage
                )
            }
            .background(ChatBotBackground())
            .navigationDestination(
                for: InputProductDetailView.Detail.self
            ) { (detail: InputProductDetailView.Detail) in
                InputProductDetailView(detail: detail)
            }
            .navigationDestination(for: [Product].self) { (products: [Product]) in
                ChatBotProductList(products: products)
                    .equatable(by: products)
            }
        }
        .background {
            GeometryReader { (p: GeometryProxy) in
                let rect: CGRect = p.frame(in: .named("rootView"))
//                let _ = print("root view minY: \(rect.minY)")
                Color.clear.task(id: rect) {
                    vm.onRootViewRectChange(rect: rect)
                }
            }
        }
        .onAppear{
            messageFieldIsFocused = true
        }
    }
}

struct ChatBotView_Previews: PreviewProvider {
    static var previews: some View {
        ChatBotView()
    }
}
