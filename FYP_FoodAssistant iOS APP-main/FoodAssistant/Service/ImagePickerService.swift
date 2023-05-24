//
//  ImagePickerService.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 16/11/2022.
//

import SwiftUI
import UIKit
import Combine

class ImagePickerService: ObservableObject {
    
    @Published var image: UIImage?
    @Published var showImagePicker = false
    
    var cancellables = Set<AnyCancellable>()
    var photo: Photo?
    
    init() {
        $image
            .sink { [weak self] (img: UIImage?) in
                guard let img else { return }
                self?.photo = Photo(image: img)
            }
            .store(in: &cancellables)
    }
    
}
