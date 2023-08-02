const true_document = parent.document;
const images = true_document.getElementsByTagName('img')
const iframes = true_document.getElementsByTagName('iframe')
const STEP_SCROLL_SIZE = 0.1

const canvas_component_id_string = 'streamlit_drawable_canvas.st_canvas'

let important_images = images

/*let already_scrolling = false;
let timer;*/

function stuff() {
    //const images = true_document.getElementsByTagName('img')
    important_images = Array.from(images).filter((img) => img.height > 200 && img.width > 200)

    if (important_images.length === 0) {
        return
    }
    // Make image scrollable and set anchor to canvas.
    for (const important_image of important_images) {
        important_image.parentElement.parentElement.style.overflow = 'scroll'
        important_image.style['transform-origin'] = 'top left'
        important_image.style['image-rendering'] = "crispEdges";

    }

    const element_with_correct_sizes = important_images[0].parentElement.parentElement
    const height = element_with_correct_sizes.clientHeight.toString()
    const width = element_with_correct_sizes.clientWidth.toString()
    const canvas_iframes = Array.from(iframes).filter((iframe) => iframe.title === canvas_component_id_string)
    for (const iframe of canvas_iframes) {
        // Set canvas iframe to correct size and make scrollable.
        iframe.height = height
        iframe.width = width
        iframe.removeAttribute('scrolling')

        // Remove gray border from canvas
        const canvas_elements = iframe.contentDocument.getElementsByTagName('canvas')
        for (const canvas_element of canvas_elements) {
            canvas_element.style.border = '0px'
            canvas_element.style['shape-rendering'] = "crispEdges";
            canvas_element.style['image-rendering'] = "crispEdges";
        }

        // Find parent node and change positioning and add the scaling.
        const canvas_root_element = iframe.contentDocument.getElementById('root')
        // Skip if iframe not completely rendered yet.
        if (canvas_root_element === undefined || canvas_root_element === null || canvas_root_element.childNodes.length === 0) {
            continue
        }

        // Skip if already adjusted.
        const canvas_parent_element = canvas_root_element.childNodes[0]
        if (canvas_parent_element.style.position === 'absolute') {
            continue
        }

        // Set canvas position for correct positioning during zooming.
        canvas_parent_element.style.position = 'absolute'
        // Add wheel+ctrl listener for zooming all images in parallel.
        canvas_parent_element.onwheel = (event) => {
            if (!event.ctrlKey) {
                return
            }
            event.preventDefault();
            let current_scale = 1.0
            // Get current zoom level of canvas if present.
            if (canvas_parent_element.style.transform !== undefined && canvas_parent_element.style.transform.length > 0) {
                const transfrom_string = canvas_parent_element.style.transform
                const scale_string = transfrom_string.substring(transfrom_string.indexOf('(') + 1, transfrom_string.indexOf(')'))
                current_scale = parseFloat(scale_string)
            }
            // Calculate new zoom level and apply to canvas and all images.
            const new_scale = current_scale + Math.sign(event.deltaY) * STEP_SCROLL_SIZE * -1
            canvas_parent_element.style.transform = `scale(${new_scale})`
            for (const important_image of important_images) {
                important_image.style.transform = `scale(${new_scale})`
            }
        }
        /*else {
                               console.log('got scrolled')
                               console.log(iframe.scrollTop)
                               for (const important_image of important_images) {
                                   important_image.scrollTop = iframe.scrollTop;
                               }
                           }*/
        /*canvas_root_element.onscroll = (event) => {
            console.log('got scrolled')
            if (!already_scrolling) {//
                already_scrolling = true;//
                clearTimeout(timer);//
                for (const important_image of important_images) {
                    important_image.scrollTop(iframe.scrollTop());
                }
                timer = setTimeout(function () {//
                    already_scrolling = false;//
                }, 100);//
            }//
        }

        iframe.parentElement.onscroll = (event) => {
            console.log(event)
        }*/
    }
}

setInterval(stuff, 1000);