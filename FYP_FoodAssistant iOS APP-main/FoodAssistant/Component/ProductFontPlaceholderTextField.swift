//
//  ProductFontPlaceholderTextField.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 16/12/2022.
//

import SwiftUI

struct ProductFontPlaceholderTextField: View {
    @Binding var text: String?
    let placeholder: String
    let keyboardType: UIKeyboardType
    let editable: Bool
    let onCommit: (() -> Void)?
    
    /// Initialize `ProductFontPlaceholderTextField` view
    ///
    /// - Parameters:
    ///   - text: a `Binding<String>` object which links to textfield's value
    ///   - placeHolder: a placeholder `String`
    ///   - keyboardType: textfield keyboard type in `UIKeyboardType` type
    ///   - onCommit: an optional closure to be triggered when user tap on `Enter` button on keyboarrd
    ///   - disabled: a boolean value indicating that if the textfield is editable or not
    init(
        text: Binding<String?>,
        placeholder: String,
        keyboardType: UIKeyboardType = .default,
        editable: Bool = true,
        onCommit: (() -> Void)? = nil
    ) {
        self._text = text
        self.placeholder = placeholder
        self.keyboardType = keyboardType
        self.editable = editable
        self.onCommit = onCommit
    }
    
    var isTextNilOrEmpty: Bool {
        text?.isEmpty ?? true
    }
    
    var body: some View {
        TextField("", text: $text, onCommit: {
            self.onCommit?()
        })
            .disabled(!editable)
            .productFont(.regular, relativeTo: .body)
            .frame(
                width: isTextNilOrEmpty ? placeholder.productAttributeWidth(.regular, relativeTo: .body) : nil,
                alignment: .leading
            )
            .background {
                Text(placeholder)
                    .productFont(.regular, relativeTo: .body)
                    .foregroundColor(.secondary)
                    .opacity(isTextNilOrEmpty ? 1 : 0)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .keyboardType(keyboardType)
    }
}
struct ProductFontPlaceholderTextField_Previews: PreviewProvider {
    @State static var text: String? = ""
    static var previews: some View {
        ProductFontPlaceholderTextField(
            text: $text,
            placeholder: "Product Barcode (e.g 4891028714842)"
        )
    }
}
