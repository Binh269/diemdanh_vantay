$(document).ready(function () {
    let chartInstance;

    function renderRooms() {
        const container = $(".rowbody");
        container.empty();

        $.post('#', { action: 'get_infor' }, function (data) {
            container.empty();

            let countHoc = 0;
            let countTrong = 0;
            let countSua = 0;

            if (data.length > 0) {
                data.forEach(function (phong) {
                    let statusClass = "";
                    switch (parseInt(phong.ttphong)) {
                        case 1: 
                            statusClass = "bg-primary";
                            countHoc++;
                            break;
                        case 2:
                            statusClass = "bg-success";
                            countTrong++;
                            break;
                        case 3: 
                            statusClass = "bg-danger";
                            countSua++;
                            break;
                        default:
                            statusClass = "bg-secondary";
                            break;
                    }

                    const roomCard = $(`
                        <div class="col-lg-6 mb-4">
                            <div class="card text-white ${statusClass} shadow">
                                <div class="card-body">
                                    <p class="m-0">${phong.tenphong}</p>
                                </div>
                            </div>
                        </div>
                    `);
                    container.append(roomCard);
                });

                $(".dang-hoc-count").text(countHoc);
                $(".dang-trong-count").text(countTrong);
                $(".dang-sua-count").text(countSua);

                updateChart([countHoc, countTrong, countSua]);
            } else {
                container.append("<p>Không có phòng học nào.</p>");
            }
        }, 'json');
    }

    function createChart(data) {
        const ctx = document.getElementById('roomChart').getContext('2d');
        chartInstance = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Hoạt động', 'Trống', 'Sửa'],
                datasets: [{
                    data: data,
                    backgroundColor: ['#4e73df', '#1cc88a', '#36b9cc'],
                    borderColor: ['#ffffff', '#ffffff', '#ffffff'],
                }],
            },
            options: {
                maintainAspectRatio: false,
                legend: {
                    display: true,
                    labels: {
                        fontStyle: 'normal',
                    },
                },
                title: {
                    display: false,
                },
            },
        });
    }

    function updateChart(data) {
        if (chartInstance) {
            chartInstance.data.datasets[0].data = data;
            chartInstance.update();
        } else {
            createChart(data);
        }
    }

    createChart([0, 0, 0]);
    renderRooms();
    function groupBy(data, key) {
        return data.reduce(function (result, item) {
            (result[item[key]] = result[item[key]] || []).push(item);
            return result;
        }, {});
    }

    function loadHongTable() {
        var container = $(".noidungtt");
        container.empty();
        container.append('<div>Đang tải dữ liệu thiết bị hỏng...</div>');

        $.post('#', { action: 'thietbi_hong' }, function (data) {
            container.empty();
            if (data.length > 0) {
                var groupedData = groupBy(data, 'maphong'); 
                for (var room in groupedData) {
                    var roomDiv = $("<div class='phong'></div>").append(`<h3>Phòng: ${room}</h3>`);
                    groupedData[room].forEach(function (thietBi) {
                        var thietBiDiv = $("<div class='thietbi'></div>");
                        thietBiDiv.append($('<div class="tt_thietbi"></div>').text(
                            `Mã Thiết Bị: ${thietBi.matb}, Tên: ${thietBi.tentb}, Trạng Thái: ${status(thietBi.tttbi)}`
                        ));
                        roomDiv.append(thietBiDiv);
                    });
                    container.append(roomDiv);
                }
            } else {
                container.append('<div>Không có thiết bị nào bị hỏng.</div>');
            }
        }, 'json');
    }

    function loadDangDungTable() {
        var container = $(".noidungtt");
        container.empty();
        container.append('<div>Đang tải dữ liệu thiết bị đang sử dụng...</div>');

        $.post('#', { action: 'thietbi_dangdung' }, function (data) {
            container.empty();
            if (data.length > 0) {
                var groupedData = groupBy(data, 'maphong'); 
                for (var room in groupedData) {
                    var roomDiv = $("<div class='phong'></div>").append(`<h3>Phòng: ${room}</h3>`);
                    groupedData[room].forEach(function (thietBi) {
                        var thietBiDiv = $("<div class='thietbi'></div>");
                        thietBiDiv.append($('<div class="tt_thietbi"></div>').text(
                            `Mã Thiết Bị: ${thietBi.matb}, Tên: ${thietBi.tentb}, Trạng Thái: ${status(thietBi.tttbi)}`
                        ));
                        roomDiv.append(thietBiDiv);
                    });
                    container.append(roomDiv);
                }
            } else {
                container.append('<div>Không có thiết bị nào đang sử dụng.</div>');
            }
        }, 'json');
    }

    function loadKhongDungTable() {
        var container = $(".noidungtt");
        container.empty();
        container.append('<div>Đang tải dữ liệu thiết bị không sử dụng...</div>');

        $.post('#', { action: 'thietbi_khongdung' }, function (data) {
            container.empty();
            if (data.length > 0) {
                var groupedData = groupBy(data, 'maphong'); 
                for (var room in groupedData) {
                    var roomDiv = $("<div class='phong'></div>").append(`<h3>Phòng: ${room}</h3>`);
                    groupedData[room].forEach(function (thietBi) {
                        var thietBiDiv = $("<div class='thietbi'></div>");
                        thietBiDiv.append($('<div class="tt_thietbi"></div>').text(
                            `Mã Thiết Bị: ${thietBi.matb}, Tên: ${thietBi.tentb}, Trạng Thái: ${status(thietBi.tttbi)}`
                        ));
                        roomDiv.append(thietBiDiv);
                    });
                    container.append(roomDiv);
                }
            } else {
                container.append('<div>Không có thiết bị nào không sử dụng.</div>');
            }
        }, 'json');
    }

    function status(tttbi) {
        switch (tttbi) {
            case 1: return "Đang sử dụng";
            case 2: return "Không sử dụng";
            case 3: return "Hỏng";
            default: return "Không xác định";
        }
    }
    $(".hong").on("click", function () {
        $('.hoatdong,.ranh').show();
        $('.hong').hide();
        loadHongTable();
    });
    $(".hoatdong").on("click", function () {
        $('.hong,.ranh').show();
        $('.hoatdong').hide();
        loadDangDungTable();
    });
    $(".ranh").on("click", function () {
        $('.hoatdong,.hong').show();
        $('.ranh').hide();
        loadKhongDungTable();
    });

    document.getElementById('exportExcel').addEventListener('click', function () {
        const rooms = document.querySelectorAll('.phong');
        const data = [];

        rooms.forEach((room) => {
            const roomName = room.querySelector('h3').textContent.replace('Phòng: ', '');
            const devices = room.querySelectorAll('.thietbi .tt_thietbi');

            devices.forEach((device) => {
                const details = device.textContent.split(', ').reduce((acc, cur) => {
                    const [key, value] = cur.split(': ');
                    acc[key.trim()] = value.trim();
                    return acc;
                }, {});

                data.push({
                    Phòng: roomName,
                    'Mã Thiết Bị': details['Mã Thiết Bị'],
                    Tên: details['Tên'],
                    'Trạng Thái': details['Trạng Thái'],
                });
            });
        });

        const worksheet = XLSX.utils.json_to_sheet(data);

        const workbook = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(workbook, worksheet, 'Thiết Bị');

        XLSX.writeFile(workbook, 'DanhSachThietBi.xlsx');
    });
});
